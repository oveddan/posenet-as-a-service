import * as posenet from '@tensorflow-models/posenet';
import {MobileNetMultiplier} from '@tensorflow-models/posenet/dist/mobilenet';

import {drawKeypoints, drawSkeleton, isMobile} from './demo_util';
import {guiState, setupGui} from './gui';

const defaultWidth = 640;
const defaultHeight = 480;

function minWidthOrValue(minWidth: number, value: number) {
  if (value <= 2) {
    return minWidth;
  } else
    return value;
}

async function setupCamera(): Promise<HTMLVideoElement> {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video') as HTMLVideoElement;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : defaultWidth,
      height: mobile ? undefined : defaultHeight
    },
  });
  video.srcObject = stream;


  return new Promise((resolve) => {
           video.onloadedmetadata = () => {
             video.width = minWidthOrValue(640, video.videoWidth);
             video.height = minWidthOrValue(480, video.videoHeight);
             resolve(video);
           };
         }) as Promise<HTMLVideoElement>;
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}


function setupWebSocket() {
  const socket = new WebSocket('ws://localhost:8080');
  guiState.socket = socket;
}

type OutputStride = 8|16|32;

// since images are being fed from a webcam
const flipHorizontal = true;

async function estimateAndRenderPoses(
    ctx: CanvasRenderingContext2D, video: HTMLVideoElement) {
  // Scale an image down to a certain factor. Too large of an image will
  // slow down the GPU
  const imageScaleFactor = guiState.input.imageScaleFactor;
  const outputStride = +guiState.input.outputStride as OutputStride;

  let minPoseConfidence: number;
  let minPartConfidence: number;

  const poses = await guiState.net.estimateMultiplePoses(
      video, imageScaleFactor, flipHorizontal, outputStride,
      guiState.poseDetection.maxPoseDetections,
      guiState.poseDetection.minPartConfidence,
      guiState.poseDetection.nmsRadius);

  minPoseConfidence = +guiState.poseDetection.minPoseConfidence;
  minPartConfidence = +guiState.poseDetection.minPartConfidence;

  const {width: videoWidth, height: videoHeight} = video;

  ctx.clearRect(0, 0, videoWidth, videoHeight);

  if (guiState.output.showVideo) {
    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-videoWidth, 0);
    ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
    ctx.restore();
  }

  const poseTime = new Date().getTime();

  const message = {
    poses,
    image: {width: videoWidth, height: videoHeight},
    poseTime
  };
  guiState.socket.send(JSON.stringify(message));
  // guiState.socket.send(JSON.stringify(poses));

  // For each pose (i.e. person) detected in an image, loop through the
  // poses and draw the resulting skeleton and keypoints if over certain
  // confidence scores
  poses.forEach(({score, keypoints}) => {
    if (score >= minPoseConfidence) {
      if (guiState.output.showPoints) {
        drawKeypoints(keypoints, minPartConfidence, ctx, guiState.output.color);
      }
      if (guiState.output.showSkeleton) {
        drawSkeleton(
            keypoints, minPartConfidence, ctx, guiState.output.lineThickness,
            guiState.output.color);
      }
    }
  });
}

const getCanvas = (): HTMLCanvasElement =>
    document.getElementById('output') as HTMLCanvasElement;

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video: HTMLVideoElement, net: posenet.PoseNet) {
  const canvas = getCanvas();
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

  const {width: videoWidth, height: videoHeight} = video;

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

    if (guiState.poseDetection.active) {
      await estimateAndRenderPoses(ctx, video);
    }

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

  setupGui(video, net, getCanvas());
  setupWebSocket();
  detectPoseInRealTime(video, net);
}

bindPage();
