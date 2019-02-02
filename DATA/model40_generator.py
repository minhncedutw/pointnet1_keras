'''
    File name: BatchGenerator for [shapenetcore_partanno_segmentation_benchmark_v0](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip)
    Author: minhnc
    Date created(MM/DD/YYYY): 8/29/2018
    Last modified(MM/DD/YYYY HH:MM): 8/29/2018 10:27 PM
    Python Version: 3.5
    Other modules: [tensorflow-gpu 1.3.0]

    Copyright = Copyright (C) 2017 of NGUYEN CONG MINH
    Credits = [None] # people who reported bug fixes, made suggestions, etc. but did not actually write the code
    License = None
    Version = 0.9.0.1
    Maintainer = [None]
    Email = minhnc.edu.tw@gmail.com
    Status = Prototype # "Prototype", "Development", or "Production"
    Code Style: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting
    Template: https://keras.io/utils/
    Reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
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
import math

import numpy as np

import keras
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator

#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================
import h5py
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def load_model40(path):
    filenames = [d for d in os.listdir(path)]
    points = None
    labels = None
    for d in filenames:
        cur_points, cur_labels = load_h5(os.path.join(path, d))
        num_points = cur_points.shape[1]
        cur_points = cur_points.reshape(1, -1, 3)
        cur_labels = cur_labels.reshape(1, -1)
        if labels is None or points is None:
            labels = cur_labels
            points = cur_points
        else:
            labels = np.hstack((labels, cur_labels))
            points = np.hstack((points, cur_points))

    points_r = points.reshape(-1, num_points, 3)
    labels_r = labels.reshape(-1, 1)
    num_labels = len(np.unique(labels_r))
    return points_r, labels_r, num_points, num_labels

class Model40Generator(Sequence):

    def __init__(self, directory, num_points=None, num_classes=None, batch_size=64, shuffle_point=False, train=True, seed=0):
        self.dir = directory
        self.num_points = num_points
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle_point = shuffle_point
        self.train = train
        self.seed = seed

        self.trn_points, self.trn_labels, self.num_points_original, _num_classes = load_model40(path=self.dir)
        if (self.num_classes is None) or (_num_classes > self.num_classes): self.num_classes = _num_classes
        self.trn_labels = keras.utils.to_categorical(y=self.trn_labels, num_classes=self.num_classes)

    def __len__(self):
        return int(np.ceil(len(self.trn_labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        left_bound = idx * self.batch_size
        right_bound = (idx + 1) * self.batch_size

        if right_bound > len(self.trn_labels):
            right_bound = len(self.trn_labels)

        batch_x = []
        batch_y = []
        for i in range(right_bound - left_bound):
            points = self.trn_points[i+left_bound]
            labels = self.trn_labels[i+left_bound]

            choice = np.random.choice(len(points), self.num_points, replace=True)
            if not self.shuffle_point: choice = np.sort(choice)
            points = points[choice, :]

            batch_x.append(points)
            batch_y.append(labels)

        return np.array(batch_x), np.array(batch_y)

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is XXXXXX Program')

    generator = ShapenetcoreSequence(directory='shapenetcore_partanno_v0', num_points=1024, class_choice='Chair', batch_size=32, train=True)
    batch_x, batch_y = generator.__getitem__(0)
    print(batch_x, batch_y)

if __name__ == '__main__':
    main()
