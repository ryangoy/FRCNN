#!usr/bin/env python

#############################
# Plug load detection pt. 1 #
#############################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# @file     detect_computers.py                 #
# @author   Ryan Goy <rgoy@berkeley.edu>        #
# @brief    Detects computers in pointgrey      #
#           images				                #
#                                               #
# @section  DESCRIPTION                         #
# Part one of the plug load detection pipeline; #
# given jpeg images from the pointgrey camera,  #
# detects computers and computes bounding boxes.#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

'''
This python script uses Ross Girshick's Faster RCNN to detect
computers in a given dataset structured as follows:
    
    Dataset
        |
        |__images
        |   |__*.jpg
        |
        |__depthmaps
        |   |__*.png
        |
        |__cameraposes.txt


Last updated: June 2016

'''

import _init_paths
from fast_rcnn.ir_test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys

#### PARAMS ####
MODEL = './models/ir/imgnet_uic_1000.caffemodel'
MODEL = '/home/ryan/vision/py-faster-rcnn/output/faster_rcnn_end2end/ir_train/vgg16_faster_rcnn_iter_28000.caffemodel'
CFG = './experiments/cfgs/faster_rcnn_end2end.yml'
GPU = 0
PROTOTXT = './models/imagenet/VGG16/faster_rcnn_end2end/test.prototxt'
WAIT = True
DATASET = 'ir'
COMP = True
VIS = False 
NUM_DETS = 100
IMDB = 'ir_test'


if __name__ == '__main__':

    # Set config and caffe parameters
    cfg_from_file(CFG)
    cfg.GPU_ID = GPU
    caffe.set_mode_gpu()
    caffe.set_device(GPU)
    
    # Initialize net
    net = caffe.Net(PROTOTXT, MODEL, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(MODEL))[0]

    # Get the dataset
    imdb = get_imdb(IMDB)
    imdb.competition_mode(True)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    
    # Call test function
    test_net(net, imdb, max_per_image=NUM_DETS, vis=VIS)
