import * as posenet from '@tensorflow-models/posenet';
import {MobileNetMultiplier} from '@tensorflow-models/posenet/dist/mobilenet';
import dat from 'dat.gui';

import {drawKeypoints, drawSkeleton} from './demo_util';

const videoWidth = 600;
const videoHeight = 500;

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

async function setupCamera(): Promise<HTMLVideoElement> {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video') as HTMLVideoElement;
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
           video.onloadedmetadata = () => {
             resolve(video);
           };
         }) as Promise<HTMLVideoElement>;
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

type GuiState = {
  input: {
    mobileNetArchitecture: '0.50'|'0.75'|'1.00'|'1.01',
    outputStride: 16,
    imageScaleFactor: 0.5,
  },
  poseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: boolean,
    showSkeleton: boolean,
    showPoints: boolean,
  },
  changeToArchitecture: boolean,
  net?: posenet.PoseNet,
  camera?: string,
  socket?: WebSocket
};

const guiState: GuiState = {
  input: {
    mobileNetArchitecture: isMobile() ? '0.50' : '0.75',
    outputStride: 16,
    imageScaleFactor: 0.5,
  },
  poseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
  },
  changeToArchitecture: false,
  net: null,
  camera: null,
  socket: null
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras, net) {
  guiState.net = net;

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const gui = new dat.GUI({width: 300});

  // The input parameters have the most effect on accuracy and speed of the
  // network
  let input = gui.addFolder('Input');
  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  const architectureController = input.add(
      guiState.input, 'mobileNetArchitecture',
      ['1.01', '1.00', '0.75', '0.50']);
  // Output stride:  Internally, this parameter affects the height and width
  // of the layers in the neural network. The lower the value of the output
  // stride the higher the accuracy but slower the speed, the higher the value
  // the faster the speed but lower the accuracy.
  input.add(guiState.input, 'outputStride', [8, 16, 32]);
  // Image scale factor: What to scale the image by before feeding it through
  // the network.
  input.add(guiState.input, 'imageScaleFactor').min(0.2).max(1.0);
  input.open();

  let multi = gui.addFolder('Multi Pose Detection');
  multi.add(guiState.poseDetection, 'maxPoseDetections').min(1).max(20).step(1);
  multi.add(guiState.poseDetection, 'minPoseConfidence', 0.0, 1.0);
  multi.add(guiState.poseDetection, 'minPartConfidence', 0.0, 1.0);
  // nms Radius: controls the minimum distance between poses that are returned
  // defaults to 20, which is probably fine for most use cases
  multi.add(guiState.poseDetection, 'nmsRadius').min(0.0).max(40.0);
  multi.open();

  let output = gui.addFolder('Output');
  output.add(guiState.output, 'showVideo');
  output.add(guiState.output, 'showSkeleton');
  output.add(guiState.output, 'showPoints');
  output.open();


  architectureController.onChange(function(architecture) {
    guiState.changeToArchitecture = architecture;
  });
}


function setupWebSocket() {
  guiState.socket = new WebSocket('ws://127.0.0.1:8080');
}

type OutputStride = 8|16|32

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
  // since images are being fed from a webcam
  const flipHorizontal = true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  async function poseDetectionFrame() {
    if (guiState.changeToArchitecture) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();

      // Load the PoseNet model weights for either the 0.50, 0.75, 1.00,
      // or 1.01 version
      guiState.net = await posenet.load(
          +guiState.changeToArchitecture as MobileNetMultiplier);

      guiState.changeToArchitecture = null;
    }

    // Scale an image down to a certain factor. Too large of an image will
    // slow down the GPU
    const imageScaleFactor = guiState.input.imageScaleFactor;
    const outputStride = +guiState.input.outputStride as OutputStride;

    let minPoseConfidence;
    let minPartConfidence;

    const poses = await guiState.net.estimateMultiplePoses(
        video, imageScaleFactor, flipHorizontal, outputStride,
        guiState.poseDetection.maxPoseDetections,
        guiState.poseDetection.minPartConfidence,
        guiState.poseDetection.nmsRadius);

    minPoseConfidence = +guiState.poseDetection.minPoseConfidence;
    minPartConfidence = +guiState.poseDetection.minPartConfidence;

    ctx.clearRect(0, 0, videoWidth, videoHeight);

    if (guiState.output.showVideo) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-videoWidth, 0);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();
    }

    guiState.socket.send(JSON.stringify(poses));

    // For each pose (i.e. person) detected in an image, loop through the
    // poses and draw the resulting skeleton and keypoints if over certain
    // confidence scores
    poses.forEach(({score, keypoints}) => {
      if (score >= minPoseConfidence) {
        if (guiState.output.showPoints) {
          drawKeypoints(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showSkeleton) {
          drawSkeleton(keypoints, minPartConfidence, ctx);
        }
      }
    });

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime
 * function.
 */
export async function bindPage() {
  // Load the PoseNet model weights with architecture 0.75
  const net = await posenet.load(0.75);

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';

  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  setupGui(video, net);
  setupWebSocket();
  detectPoseInRealTime(video, net);
}

bindPage();
