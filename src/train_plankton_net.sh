#!/usr/bin/env sh

TOOLS=./../../caffe/build/tools

$TOOLS/caffe train \
  --solver=plankton_solver.prototxt
