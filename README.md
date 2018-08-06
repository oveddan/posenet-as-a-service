# Posenet as a Service

This is a **very early Work in Progress.**

This repository provides a way to use PoseNet as a service.  It does human pose estimation
using an RGB Image, then sends the estimated poses to all subscribers, allowing the pose data
to be consumed in other programs either on the same machine or across a network.

It contains a:
* In-browser application and bundler to estimate poses from a video
* Minimal node.js server that relayes estimated poses from the browser to any subscribing clients via websocket.  In the future it will support OSC and streaming protocols.

The client-side app is based off of code from the PoseNet camera demo.

## Instructions:

Install dependencies:

    yarn

Start the node.js server and client (parcel) bundler:

    yarn start

You can then open a browser window and start estimating poses, by going to http://localhost:1234; the poses wiil be sent from the client to the node server via websocket, and the server will broadcast that to the rest of the clients.  The server sends and receives messages on port 8080.

In TouchDesigner, a WebSocketDAT can be used to receive poses.  Make sure the it is connecting to host: localhost and port: 8080

