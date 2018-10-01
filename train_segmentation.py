'''
    File name: Train Pointnet Segmentation
    Author: minhnc
    Date created(MM/DD/YYYY): 10/1/2018
    Last modified(MM/DD/YYYY HH:MM): 10/1/2018 8:30 AM
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

import keras
from keras import optimizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Reshape
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda, concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard, RemoteMonitor, ReduceLROnPlateau
import tensorflow as tf

#==============================================================================
# Constant Definitions
#==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./DATA/shapenetcore_partanno_segmentation_benchmark_v0', help='data directory')
parser.add_argument('--chose_cat', action='append', default='Airplane', help='Add category choice')
parser.add_argument('--num_points', type=int, default=2048, help='number of input points')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--num_epoches', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--trained_model', type=str, default='',  help='model path')
opt = parser.parse_args()
print(opt)

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

def callbacks(batch_size=None):
    callbacks_list = [
        ModelCheckpoint(
                        filepath='./outputs/model_checkpoints/model.loss.{epoch:02d}.hdf5', # string, path to save the model file.
                        monitor='val_loss', # quantity to monitor.
                        save_best_only=True, # if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
                        mode='auto', # one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity.
                        save_weights_only='false', # if True, then only the model's weights will be saved (model.save_weights(filepath)), else the full model is saved (model.save(filepath)).
                        period=1, # Interval (number of epochs) between checkpoints.
                        verbose=1), # verbosity mode, 0 or 1.
        TensorBoard(log_dir='./outputs/graph',
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True,
                    embeddings_freq=0,
                    embeddings_layer_names=None,
                    embeddings_metadata=None,
                    embeddings_data=None),
        # RemoteMonitor(root='http://localhost:9000',
        #               path='./outputs/monitor/',
        #               field='data', headers=None,
        #               send_as_json=False),
        # ReduceLROnPlateau(monitor='val_loss', # quantity to be monitored.
        #                   factor=0.2, # factor by which the learning rate will be reduced. new_lr = lr * factor
        #                   patience=4, # number of epochs with no improvement after which learning rate will be reduced.
        #                   verbose=1, # int. 0: quiet, 1: update messages.
        #                   mode='min', # In min mode, lr will be reduced when the quantity monitored has stopped decreasing;
        #                               # in max mode it will be reduced when the quantity monitored has stopped increasing;
        #                               # in auto mode, the direction is automatically inferred from the name of the monitored quantity.
        #                   min_lr=1e-6, # lower bound on the learning rate.
        #                   cooldown=2), # number of epochs to wait before resuming normal operation after lr has been reduced.
    ]
    return callbacks_list

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is Train Pointnet Segmentation Program')

    '''
    Define network/training parameters
    '''
    num_points = opt.num_points
    num_epoches = opt.num_epoches
    batch_size = opt.batch_size
    cat_choices = opt.chose_cat

    '''
    Define dataset/data-loader
    '''
    # from DATA.shapenet_generator1 import ShapenetGenerator
    # train_generator = ShapenetGenerator(directory='./DATA/shapenetcore_partanno_v0', num_points=1024, class_choice='Chair', batch_size=8, train=True)
    # valid_generator = ShapenetGenerator(directory='./DATA/shapenetcore_partanno_v0', num_points=1024, class_choice='Chair', batch_size=8, train=False)
    # num_classes = 4
    from DATA.shapenet_generator2 import ShapenetGenerator
    trn_generator = ShapenetGenerator(directory=opt.data_dir, num_points=num_points, cat_choices=cat_choices, batch_size=batch_size, train=True)
    val_generator = ShapenetGenerator(directory=opt.data_dir, num_points=num_points, cat_choices=cat_choices, batch_size=batch_size, train=False)
    num_classes = 5

    '''
    Define model
    '''
    model = PointNet(num_points=num_points, num_classes=num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    callback_list = callbacks(batch_size=batch_size)

    '''
    Train then Evaluate model
    '''
    # train model
    trn_his = model.fit_generator(generator=trn_generator, validation_data=val_generator, epochs=num_epoches, verbose=1)

    # evaluate model
    score = model.evaluate_generator(generator=val_generator, verbose=1)


if __name__ == '__main__':
    main()
