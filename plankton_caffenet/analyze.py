import numpy as np
import sys

# hack sys.path so we can import caffe
caffe_python_path = '~/caffe/python'
sys.path.insert(0, caffe_python_path)

# Set the default GPU that caffe will use.
import caffe
caffe.set_mode_gpu()

net = caffe.Net('plankton_deploy.prototxt', caffe.TRAIN)

net1 = caffe.Net('../scripts/net_surgery_new_deploy.prototxt', caffe.TRAIN)
print "HIDDEN SIZES"
params = net.blobs.keys()
for pr in params:
  print pr, ':', net.blobs[pr].data.shape

print "WEIGHTS"
params = net.params.keys()
for pr in params:
  print pr, ':', net.params[pr][0].data.shape

print '---------'

print "HIDDEN SIZES"
params = net1.blobs.keys()
for pr in params:
  print pr, ':', net1.blobs[pr].data.shape

print "WEIGHTS"
params = net1.params.keys()
for pr in params:
  print pr, ':', net1.params[pr][0].data.shape