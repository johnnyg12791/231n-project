from scipy.misc import imread, imresize
import numpy as np
import os
import sys
import math

BATCH_SIZE = 200

# hack sys.path so we can import caffe
caffe_python_path = '~/caffe/python'
sys.path.insert(0, caffe_python_path)

# Set the default GPU that caffe will use.
import caffe
#caffe.set_device(0)
caffe.set_mode_gpu()

model_file = '../plankton_caffenet/plankton_deploy.prototxt'
weights_file = '../plankton_caffenet/snapshot_iter_4501.caffemodel'
net = caffe.Net(model_file, weights_file, caffe.TEST)

directory = '../data/test'
outputFile = open('../data/submission.csv', 'w')

images = [item for item in os.listdir(directory)]
num_images = len(images)
num_batches = int(math.ceil(num_images / BATCH_SIZE))

input_data = np.zeros((len(images), 1, 227, 227))
for batch_num in range(num_batches):
  print "Batch:", batch_num, "/", num_batches
  for i in range(BATCH_SIZE):
    img_index = i + BATCH_SIZE * batch_num
    if (img_index >= num_images): break
    image = images[img_index]
    
    #if image == '.DS_Store': continue
    img = imread('../data/test/' + image)
  
    # Resize image to same size as mean
    #H_mean, W_mean = mean.shape[1:]
    # TODO
    H_mean,W_mean = 256, 256
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
    net.blobs['data'].data[i] = img

  # Call net forward
  net.forward()
  
  # Pull out the probabilities from the network and print them
  probs = net.blobs['prob'].data
  for i in range(BATCH_SIZE):
   img_index = i + BATCH_SIZE * batch_num
   if (img_index >= num_images): break
   image = images[img_index]
   outputFile.write(image +',' + ','.join(['%.16f' % num for num in probs[i]]) + '\n')

outputFile.close()