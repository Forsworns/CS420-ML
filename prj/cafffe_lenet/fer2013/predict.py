#coding = utf-8
import numpy as np
import sys, os
import cv2
import caffe

#caffe.set_device(0)
#caffe.set_mode_gpu()

net_file = './lenet_deploy.prototxt'
caffe_model = './models/lenet_iter_2000.caffemodel'

# load model
net = caffe.Net(net_file, caffe_model, caffe.TEST)

src_im = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
src_im = src_im.reshape((1, 1, 48, 48))
net.blobs['data'].data[...] = src_im
out = net.forward()
print out
