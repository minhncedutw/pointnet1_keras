'''
    File name: Test Segmentation
    Author: minhnc
    Date created(MM/DD/YYYY): 10/2/2018
    Last modified(MM/DD/YYYY HH:MM): 10/2/2018 5:22 AM
    Python Version: 3.6
    Other modules: [None]

    Copyright = Copyright (C) 2017 of NGUYEN CONG MINH
    Credits = [None] # people who reported bug fixes, made suggestions, etc. but did not actually write the code
    License = None
    Version = 0.9.0.1
    Maintainer = [None]
    Email = minhnc.edu.tw@gmail.com
    Status = Prototype # "Prototype", "Development", or "Production"
    Code Style: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting
'''

#==============================================================================
# Imported Modules
#==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # The GPU id to use, usually either "0" or "1"

import numpy as np

from pointnet import PointNet

from points_visualization import visualize

#==============================================================================
# Constant Definitions
#==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./DATA/shapenetcore_partanno_segmentation_benchmark_v0', help='data directory')
parser.add_argument('--chose_cat', action='append', default='Airplane', help='Add category choice')
parser.add_argument('--chose_idx', action='append', default='0', help='Add index of test choice')
parser.add_argument('--num_points', type=int, default=4096, help='number of input points') # 8192 16384 32768
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--num_epoches', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--pretrained_model', type=str, default='./outputs/checkpoints/model.loss.47.hdf5',  help='model path')
opt = parser.parse_args()
print(opt)

#==============================================================================
# Function Definitions
#==============================================================================

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is XXXXXX Program')

    '''
    Define network/training parameters
    '''
    num_points = opt.num_points
    batch_size = opt.batch_size
    cat_choices = opt.chose_cat
    weight_path = opt.pretrained_model
    idx_choices = opt.chose_idx

    '''
    Define dataset/data-loader
    '''
    from DATA.shapenet_generator2 import ShapenetGenerator
    test_generator = ShapenetGenerator(directory=opt.data_dir, num_points=num_points, cat_choices=cat_choices,
                                      batch_size=1, shuffle=False, train=False)
    num_classes = 5

    '''
    Define model and Load weights
    '''
    model = PointNet(num_points=num_points, num_classes=num_classes)
    model.load_weights(filepath=weight_path)

    '''
    Predict result
    '''
    batch_x, batch_y = test_generator.__getitem__(0)
    pred = model.predict(x=batch_x)
    pred = np.squeeze(pred) # remove dimension of batch
    points = np.squeeze(batch_x) # remove dimension of batch
    pred_label = pred.argmax(axis=1)

    # better way display
    visualize(x=points[:, 0], y=points[:, 1], z=points[:, 2], label=pred_label)

if __name__ == '__main__':
    main()
