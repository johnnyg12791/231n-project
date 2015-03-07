#!/usr/bin/env sh

CAFFE=../../caffe
TOOLS=./../../caffe/build/tools

$TOOLS/caffe train \
  --solver=plankton_solver.prototxt \
  --weights $CAFFE/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
  -gpu 0