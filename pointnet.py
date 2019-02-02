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
from keras.layers import Convolution1D, BatchNormalization, MaxPooling1D, GlobalMaxPooling1D
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

def extend_dimension(global_feature, axis):
    '''
    Extend dimension of a tensor(example: [None, 1024] to [None, 1, 1024])
    :param global_feature:
    :param axis:
    :return:
    '''
    return tf.expand_dims(global_feature, axis)

def extend_size(global_feature, num_points):
    '''
    Extend size of a tensor(example: [None, 1, 1024] to [None, num_points, 1024])
    :param global_feature:
    :param num_points:
    :return:
    '''
    return tf.tile(global_feature, [1, num_points, 1])

def multilayer_perceptron(inputs, mlp_nodes):
    '''
    Define multilayer-perceptron
    :param inputs: a tensor of input data
    :param layer_nodes: an array of intergers that defines num-nodes for each layer(example: [16, 16, 32, 32, 64], ...)
    :return: outputs of each layer
    '''
    mlp = []
    x = inputs
    for i, num_nodes in enumerate(mlp_nodes):
        x = Convolution1D(filters=num_nodes, kernel_size=1, activation='relu')(x)
        x = BatchNormalization()(x)
        mlp.append(x)
    return mlp


def TNet(inputs, tsize, mlp_nodes=(64, 128, 1024), fc_nodes=(512, 256)):
    '''
    Define T-Net(joint aligment network) to predict affine transformation matrix
    :param inputs: a tensor of input data
    :param tsize: an integer that defines the size of transformation matrix
    :param mlp_nodes: an array of intergers that defines num-nodes for each layer(example: [16, 16, 32, 32, 64], ...)
    :param fc_nodes:an array of intergers that defines num-nodes for each layer(example: [16, 16, 32, 32, 64], ...)
    :return:
    '''
    x = inputs
    for i, num_nodes in enumerate(mlp_nodes):
        x = Convolution1D(filters=num_nodes, kernel_size=1, activation='relu')(x)
        x = BatchNormalization()(x)

    x = GlobalMaxPooling1D()(x)

    for i, num_nodes in enumerate(fc_nodes):
        x = Dense(num_nodes, activation='relu')(x)
        x = BatchNormalization()(x)

    x = Dense(tsize*tsize, weights=[np.zeros([num_nodes, tsize*tsize]), np.eye(tsize).flatten().astype(np.float32)])(x) # constrain the feature transformation matrix to be close to orthogonal matrix
    transformation_matrix = Reshape((tsize, tsize))(x)
    return transformation_matrix


def PointNetFull(num_points, num_classes, type='seg'):
    '''
    Pointnet full architecture
    :param num_points: an integer that is the number of input points
    :param num_classes: an integer that is number of categories
    :param type: a string of 'seg' or 'cls' to select Segmentation network or Classification network
    :return:
    '''

    inputs = Input(shape=(num_points, 3))

    ''' 
    Begin defining Pointnet Architecture
    '''
    tnet1 = TNet(inputs=inputs, tsize=3, mlp_nodes=(128, 128, 1024), fc_nodes=(512, 256))
    aligned_feature1 = keras.layers.dot(inputs=[inputs, tnet1], axes=2)

    extracted_feature11, extracted_feature12, extracted_feature13 = multilayer_perceptron(inputs=aligned_feature1,
                                                                                          mlp_nodes=(64, 128, 128))

    tnet2 = TNet(inputs=inputs, tsize=128, mlp_nodes=(128, 128, 1024), fc_nodes=(512, 256))
    aligned_feature2 = keras.layers.dot(inputs=[extracted_feature13, tnet2], axes=2)

    extracted_feature21, extracted_feature22 = multilayer_perceptron(inputs=aligned_feature2, mlp_nodes=(512, 2048))

    global_feature = GlobalMaxPooling1D()(extracted_feature22)

    global_feature_seg = Lambda(extend_dimension, arguments={'axis': 1})(global_feature)
    global_feature_seg = Lambda(extend_size, arguments={'num_points': num_points})(global_feature_seg)

    # Classification block
    cls = Dense(512, activation='relu')(global_feature)
    cls = BatchNormalization()(cls)
    cls = Dense(256, activation='relu')(cls)
    cls = BatchNormalization()(cls)
    cls = Dense(num_classes, activation='softmax')(cls)
    cls = BatchNormalization()(cls)

    # Segmentation block
    seg = concatenate([extracted_feature11, extracted_feature12, extracted_feature13, aligned_feature2, extracted_feature21, global_feature_seg])
    _, _, seg  = multilayer_perceptron(inputs=seg, mlp_nodes=(256, 256, 128))
    seg = Convolution1D(num_classes, 1, padding='same', activation='softmax')(seg)
    ''' 
    End defining Pointnet Architecture
    '''

    if type=='seg':
        model = Model(inputs=inputs, outputs=seg)
    elif type=='cls':
        model = Model(inputs=inputs, outputs=cls)
    else:
        raise ValueError("ERROR!!! 'type' must be 'seg' or 'cls'")
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
