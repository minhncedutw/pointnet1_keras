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

import numpy as np

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

class ShapenetcoreSequence(Sequence):

    def __init__(self, directory, num_points, class_choice, batch_size=32, train=True):
        self.dir = directory
        self.num_points = num_points
        self.class_choice = class_choice
        self.batch_size = batch_size

        self.cat_file = os.path.join(self.dir, 'synsetoffset2category.txt') # category file
        self.cat_dict = {} # category dictionary

        with open(self.cat_file, 'r') as f:
            for line in f:
                [cat, folder] = line.strip().split()
                self.cat_dict[cat] = folder # 'category' information is saved in 'folder'

        self.cur_folder = os.path.join(self.dir, self.cat_dict[class_choice]) # current folder
        self.points_path = os.path.join(self.cur_folder, "points")
        self.labels_path = os.path.join(self.cur_folder, "points_label")

        self.points_filenames = [file for file in sorted(os.listdir(self.points_path))]
        if train:
            self.points_filenames = self.points_filenames[:int(len(self.points_filenames) * 0.9)]
        else:
            self.points_filenames = self.points_filenames[int(len(self.points_filenames) * 0.9):]

        self.names = []
        for file in self.points_filenames:
            self.names.append(file.split('.')[0])

        self.labels = [label for label in os.listdir(self.labels_path)]

    def __len__(self):
        return int(np.ceil(len(self.names) / float(self.batch_size)))

    def __getitem__(self, idx):
        left_bound = idx * self.batch_size
        right_bound = (idx + 1) * self.batch_size

        if right_bound > len(self.names):
            right_bound = len(self.names)

        batch_x = []
        batch_y = []
        for i in range(right_bound - left_bound):
            cur_points_path = os.path.join(self.points_path, self.points_filenames[left_bound + i])
            cur_points = np.loadtxt(cur_points_path).astype('float32')

            sub_batch = []
            for label_idx in range(len(self.labels)):
                label_filename = self.names[left_bound + i] + '.seg'
                label_path = os.path.join(self.labels_path, self.labels[label_idx], label_filename)

                if os.path.isfile(label_path):
                    cur_label = np.loadtxt(label_path).astype('float32')
                else:
                    cur_label = np.zeros((len(cur_points)))
                sub_batch.append(cur_label)
            sub_batch = np.array(sub_batch).transpose(1, 0)

            choice = np.random.choice(len(cur_points), self.num_points, replace=True)
            cur_points = cur_points[choice, :]
            sub_batch = sub_batch[choice, :]

            batch_x.append(cur_points)
            batch_y.append(sub_batch)

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
