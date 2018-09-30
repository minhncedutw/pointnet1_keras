'''
    File name: BatchGenerator
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

# datagen = ImageDataGenerator
# generator = datagen.flow_from_directory()

#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================
from skimage.io import imread
from skimage.transform import resize

class ShapenetGenerator(Sequence):

    def __init__(self, directory, num_points, cat_choices=None, num_classes=None, batch_size=32, shuffle=False, train=True):
        self.dir = directory
        self.num_points = num_points
        self.num_classes = num_classes
        self.cat_choices = cat_choices
        self.batch_size = batch_size

        self.cat_list_file = os.path.join(self.dir, 'synsetoffset2category.txt') # category file
        self.cat_dict = {} # category dictionary

        # create dict of category-path
        with open(self.cat_list_file, 'r') as f: # get list of category/path in the cat_list_file
            for line in f:
                [category, path] = line.strip().split()
                self.cat_dict[category] = path # 'category' information is saved in 'folder'
        if not cat_choices is None: # exclude category/path that are not chosen
            self.cat_dict = {category: path for category, path in self.cat_dict.items() if category in cat_choices}

        self.datapath = []
        for item in self.cat_dict:
            points_path = os.path.join(self.dir, self.cat_dict[item], "points") # path to points folder
            labels_path = os.path.join(self.dir, self.cat_dict[item], "points_label") # path to labels folder

            self.points_filenames = [file for file in sorted(os.listdir(points_path))]
            if shuffle:
                np.random.shuffle(self.points_filenames)
            if train:
                self.points_filenames = self.points_filenames[:int(len(self.points_filenames) * 0.9)]
            else:
                self.points_filenames = self.points_filenames[int(len(self.points_filenames) * 0.9):]

            for fn in self.points_filenames:
                token = (os.path.splitext(os.path.basename(fn))[0])
                pts_file = os.path.join(points_path, token + '.pts')
                seg_file = os.path.join(labels_path, token + '.seg')
                self.datapath.append((item, pts_file, seg_file))

        count_classes = 0
        for i in range(math.ceil(len(self.datapath) / 50)):
            biggest_label = np.max(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
            if biggest_label > count_classes:
                count_classes = biggest_label
        if (self.num_classes is None) or (self.num_classes < count_classes + 1):
            self.num_classes = count_classes + 1

    def __len__(self):
        return int(np.ceil(len(self.datapath) / float(self.batch_size)))

    def __getitem__(self, idx):
        left_bound = idx * self.batch_size
        right_bound = (idx + 1) * self.batch_size

        if right_bound > len(self.datapath):
            right_bound = len(self.datapath)

        batch_x = []
        batch_y = []
        for i in range(right_bound - left_bound):
            points = np.loadtxt(self.datapath[left_bound + i][1]).astype('float32')
            labels = np.loadtxt(self.datapath[left_bound + i][2]).astype('int')

            choice = np.random.choice(len(points), self.num_points, replace=True)
            points = points[choice, :]
            labels = labels[choice]

            onehot_labels = keras.utils.to_categorical(y=labels, num_classes=self.num_classes)

            batch_x.append(points)
            batch_y.append(onehot_labels)

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
