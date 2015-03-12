import numpy as np
import sys

# hack sys.path so we can import caffe
caffe_python_path = '~/caffe/python'
sys.path.insert(0, caffe_python_path)

# Set the default GPU that caffe will use.
import caffe
caffe.set_mode_gpu()

net = caffe.Net('../../caffe/models/bvlc_reference_caffenet/deploy.prototxt', '../../caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', caffe.TRAIN)
new_net = caffe.Net('net_surgery_new_deploy.prototxt', caffe.TRAIN)

params = net.params.keys()
params.remove('conv1')
for pr in params:
  new_net.params[pr] = net.params[pr]

new_net.params['conv1'][0].data[...] = np.average(net.params['conv1'][0].data, axis=1).reshape((96, 1, 11, 11))
new_net.params['conv1'][1].data[...] = net.params['conv1'][1].data

print "First layer now has dimension:", new_net.params['conv1'][0].data.shape
new_net.save('../data/bvlc_reference_caffenet_surgery.caffemodel')