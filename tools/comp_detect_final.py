#############################
# Computer detection script #
#############################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Adapted from Ross         #
# Girshick's test_net.py    #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys

#### PARAMS ####
MODEL = './models/ir/imgnet_uic_1000.caffemodel'
#MODEL = './data/ir_models/ir_office.caffemodel'
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
