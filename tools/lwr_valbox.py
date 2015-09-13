# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 13:38:17 2015

@author: liuwr
"""
import ipdb
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
#import ipdb:

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args                     
                     
if __name__ == '__main__':
    args = parse_args()
    
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'test.prototxt')  
                                
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)  

    # Load pre-computed Selected Search object proposals
    testset = sio.loadmat(\
            '/home/liuwr/rcnn/data/selective_search_data/voc_2007_test.mat')
    maxioubox = sio.loadmat(\
            '/home/liuwr/rcnn/MaxIOUofDataSetBySS/VOC2007/test/voc2007maxIOU.mat')
    maxIOUmat = maxioubox['maxIOUmat']
    maxIOUmat = maxIOUmat[0]
    image = testset['images']
    proposal_box = testset['boxes']
    proposal_box = proposal_box[0]
    imagesum = len(image)
    result = []
    
    from IPython import embed
    embed()

    for i in range(imagesum):
        # Load the image
        im_file = '/data/VOC2007/VOCdevkit/VOC2007/JPEGImages/' \
                    + str(image[i][0][0]) + '.jpg'
        im = cv2.imread(im_file)

        # Detect all object classes and regress object bounds
        #timer = Timer()
        #timer.tic()
        obj_proposals = proposal_box[i]
        boxindex = maxIOUmat[i][0][0]
        object_num = len(boxindex)
        
	ipdb.set_trace()

        for j in range(object_num):
            ioumaxbox = obj_proposals[boxindex[j]-1]
            ioumaxbox = np.array([ioumaxbox[1],ioumaxbox[0],ioumaxbox[3],ioumaxbox[2]])
            ioumaxbox.shape = (4,1)
            ioumaxbox = np.transpose(ioumaxbox)
            scores, boxes = im_detect(net, im, ioumaxbox)
            tempscore = scores[0,1:21]
            scoreEcpBack = sorted(tempscore,reverse = True)
            scoreindex = range(len(tempscore))
            scoreindex.sort(key = lambda x: -tempscore[x])
            maxindex = scoreindex[0]
            result.append((str(image[i][0][0]),scoreindex[0:5],boxes[0,maxindex*4:(maxindex+1)*4]))
            
            
        # Visualize detections for each class
        #CONF_THRESH = 0.8
        #NMS_THRESH = 0.3
        #classes = ('car',)
        #for cls in classes:
        #cls_ind = CLASSES.index(cls)
        #cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        #cls_scores = scores[:, cls_ind]
        #keep = np.where(cls_scores >= CONF_THRESH)[0]
        #cls_boxes = cls_boxes[keep, :]
        #cls_scores = cls_scores[keep]
        #dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
        #keep = nms(dets, NMS_THRESH)
        #dets = dets[keep, :]
        #print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,CONF_THRESH)
        # vis_detections(im, cls, dets, thresh=CONF_THRESH)
