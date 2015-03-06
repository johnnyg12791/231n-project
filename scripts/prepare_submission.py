# TODO

from scipy.misc import imread, imresize
import numpy as np
import os
import sys

# hack sys.path so we can import caffe
caffe_python_path = '~/caffe/python'
sys.path.insert(0, caffe_python_path)

# Set the default GPU that caffe will use.
import caffe
caffe.set_device(0)

model_file = '../plankton_caffenet/plankton_deploy.prototxt'
weights_file = '../plankton_caffenet/snapshot_iter_4501.caffemodel'
net = caffe.Net(model_file, weights_file, caffe.TEST)

directory = '../data/test'
outputFile = open('../data/submission.csv', 'w')

images = [item for item in os.listdir(directory)]
images = images[1:100]
input_data = np.zeros((len(images), 1, 227, 227))
for i,image in enumerate(images):
  if image == '.DS_Store': continue
  print i
  img = imread('../data/test/' + image)
  
  # Before feeding the image to the network, we need to preprocess it:
  # 1) Resize image to (256, 256)
  # 2) Swap channels from RGB to BGR (for CaffeNet)
  # 3) Reshape from (H, W, K) to (K, H, W)
  # 4) Subtract ImageNet mean
  # 5) Crop or resize to (227, 227)
  
  # Resize image to same size as mean
  #H_mean, W_mean = mean.shape[1:]
  # TODO
  H_mean = 256
  W_mean = 256
  img = imresize(img, (H_mean, W_mean))
  
  # Reshape from (H, W, K) to (K, H, W)
  #img = img.transpose(2, 0, 1)
  
  # Subtract mean
  #img = img - mean
  
  # Crop to input size of network
  H_in, W_in = net.blobs['data'].data.shape[2:]
  H0 = (H_mean - H_in) / 2
  H1 = H0 + H_in
  W0 = (W_mean - W_in) / 2
  W1 = W0 + W_in
  img = img[H0:H1, W0:W1]
  
  # Copy input data to input blob of the network
  input_data[i] = img
  
print net.blobs['data'].data.shape
net.blobs['data'].data[0] = input_data
  
  # Call net forward
net.forward()
  
  # Pull out the probabilities from the network and print them
probs = net.blobs['prob'].data[0]
print probs[0:1]
#outputFile.write(image +',' + ','.join(['%.16f' % num for num in probs]) + '\n')