'''
    File name: PointNet Definition
    Author: minhnc
    Date created(MM/DD/YYYY): 10/2/2018
    Last modified(MM/DD/YYYY HH:MM): 10/2/2018 5:25 AM
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

import keras
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Reshape
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda, concatenate
import tensorflow as tf

#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================
def exp_dim(global_feature, num_points):
    return tf.tile(global_feature, [1, num_points, 1])

def PointNet(num_points, num_classes):
    '''
        inputs:
            num_points: integer > 0, number of points for each point cloud image
            num_classes: total numbers of segmented classes
        outputs:
            onehot encoded array of classified points
    '''
    '''
    Begin defining Pointnet Architecture
    '''
    input_points = Input(shape=(num_points, 3))

    x = Convolution1D(64, 1, activation='relu',
                      input_shape=(num_points, 3))(input_points)
    x = BatchNormalization()(x)
    x = Convolution1D(128, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=num_points)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    input_T = Reshape((3, 3))(x)

    ## forward net
    g = keras.layers.dot(inputs=[input_points, input_T], axes=2)
    g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)

    ## feature transformation net
    f = Convolution1D(64, 1, activation='relu')(g)
    f = BatchNormalization()(f)
    f = Convolution1D(128, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Convolution1D(1024, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = MaxPooling1D(pool_size=num_points)(f)
    f = Dense(512, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(256, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = Reshape((64, 64))(f)

    ## forward net
    g = keras.layers.dot(inputs=[g, feature_T], axes=2)
    seg_part1 = g
    g = Convolution1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(128, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(1024, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    ## global_feature
    global_feature = MaxPooling1D(pool_size=num_points)(g)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)

    ## point_net_seg
    c = concatenate([seg_part1, global_feature])
    c = Convolution1D(512, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(256, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    prediction = Convolution1D(num_classes, 1, activation='softmax')(c)
    ''' 
    End defining Pointnet Architecture
    '''

    ''' 
    Define Model
    '''
    model = Model(inputs=input_points, outputs=prediction)
    print(model.summary())

    return model

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is PointNet Definition Program')

    if argv is None:
        argv = sys.argv

    if len(argv) > 1:
        for i in range(len(argv) - 1):
            print(argv[i + 1])


if __name__ == '__main__':
    main()
