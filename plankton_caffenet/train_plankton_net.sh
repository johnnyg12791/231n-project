#!/usr/bin/env sh

CAFFE=../../caffe
TOOLS=./../../caffe/build/tools

$TOOLS/caffe train \
  --solver=plankton_solver.prototxt \
  --weights ../data/bvlc_reference_caffenet_surgery.caffemodel \
  -gpu 0
