import * as posenet from '@tensorflow-models/posenet';
import {GUI} from 'dat.gui';

import {isMobile} from './demo_util';

export type GuiState = {
  input: {
    mobileNetArchitecture: '0.50'|'0.75'|'1.00'|'1.01',
    outputStride: 16,
    imageScaleFactor: 0.5,
  },
  poseDetection: {
    active: boolean,
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: boolean,
    showSkeleton: boolean,
    showPoints: boolean,
    color: string,
    lineThickness: number,
  },
  changeToArchitecture: boolean,
  net?: posenet.PoseNet,
  camera?: string,
  socket?: WebSocket
};

export const guiState: GuiState = {
  input: {
    mobileNetArchitecture: isMobile() ? '0.50' : '0.75',
    outputStride: 16,
    imageScaleFactor: 0.5,
  },
  poseDetection: {
    active: true,
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    color: '#00FFFF',
    lineThickness: 5
  },
  changeToArchitecture: false,
  net: null,
  camera: null,
  socket: null
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
export function setupGui(cameras, net) {
  guiState.net = net;

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const gui = new GUI({width: 300});

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


  let multi = gui.addFolder('Pose Detection');
  multi.add(guiState.poseDetection, 'active');
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
  output.addColor(guiState.output, 'color');
  output.add(guiState.output, 'lineThickness', 1, 10);
  output.open();


  architectureController.onChange(function(architecture) {
    guiState.changeToArchitecture = architecture;
  });
}
