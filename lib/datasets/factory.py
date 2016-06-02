# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.ir import ir
import numpy as np
import datasets.imagenet

##### FOR NEW DATASETS CHANGE DEVKIT PATH #####
ir_devkit_path = '/home/ryan/datasets/ir_dataset/'
imagenet_devkit_path = '/home/ryan/datasets/computer_dataset/'

for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = '/media/VSlab2/imagenet/ILSVRC13'
    __sets[name] = (lambda split=split, devkit_path=devkit_path:datasets.imagenet.imagenet(split,devkit_path))

for split in ['train', 'test']:
    name = 'ir_{}'.format(split)
    __sets[name] = (lambda split=split: ir(split, devkit_path=ir_devkit_path))

for split in ['train', 'test']:
    name = 'desktops_{}'.format(split)
    __sets[name] = (lambda split=split: imagenet(split, devkit_path=imagenet_devkit_path))
# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
