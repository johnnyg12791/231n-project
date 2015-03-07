#!/usr/bin/env sh

CAFFE=../../caffe
TOOLS=./../../caffe/build/tools

$TOOLS/caffe train \
  --solver=plankton_solver.prototxt \
  --weights lenet_iter_10000.caffemodel
  -gpu 0 # TODO: check this