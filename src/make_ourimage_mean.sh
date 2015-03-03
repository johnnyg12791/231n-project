#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

./../../caffe/build/tools/compute_image_mean ../demo/plankton_train_lmdb \
  ../demo/plankton_mean.binaryproto

echo "Done."