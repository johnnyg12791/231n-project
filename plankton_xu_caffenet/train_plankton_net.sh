#!/usr/bin/env sh

CAFFE=../../caffe
TOOLS=./../../caffe/build/tools

$TOOLS/caffe train \
  --solver=plankton_solver.prototxt \
  -gpu 0
