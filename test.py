from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # The GPU id to use, usually either "0" or "1"

import numpy as np


#==============================================================================
# Constant Definitions
#==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./DATA/shapenetcore_partanno_segmentation_benchmark_v0', help='data directory')
parser.add_argument('--chose_cat', action='append', default='Airplane', help='Add category choice')
parser.add_argument('--num_points', type=int, default=4096, help='number of input points') # 8192 16384 32768
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--num_epoches', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--trained_model', type=str, default='',  help='model path')
opt = parser.parse_args()
print(opt)

num_points = opt.num_points
num_epoches = opt.num_epoches
batch_size = opt.batch_size
cat_choices = opt.chose_cat

from DATA.arlab_generator import ShapenetGenerator
trn_generator = ShapenetGenerator(directory=opt.data_dir, num_points=num_points, cat_choices=cat_choices, batch_size=batch_size, train=True)
val_generator = ShapenetGenerator(directory=opt.data_dir, num_points=num_points, cat_choices=cat_choices, batch_size=batch_size, train=False)
num_classes = 5

