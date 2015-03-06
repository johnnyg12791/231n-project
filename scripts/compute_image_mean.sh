#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

./../../caffe/build/tools/compute_image_mean ../data/plankton_train_lmdb \
  ../data/plankton_mean.binaryproto

echo "Done."
