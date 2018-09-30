# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 18:48:18 2018

@author: MinhNC
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import keras
import keras.backend as K
import tensorflow as tf
from keras import optimizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Reshape
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda, concatenate
#from keras.utils import np_utils
import h5py


def mat_mul(A, B):
    return tf.matmul(A, B)


def exp_dim(global_feature, num_points):
    return tf.tile(global_feature, [1, num_points, 1])


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def rotate_point_cloud(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data

'''
global variable
'''
# number of points in each sample
num_points = 1024
# num_points = 1048576
# number of categories
k = 4
# epoch number
epo = 10
# define optimizer
adam = optimizers.Adam(lr=0.001, decay=0.7)

'''
Pointnet Architecture
'''
# input_Transformation_net
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

# forward net
# g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
g = keras.layers.dot(inputs=[input_points, input_T], axes=2)
g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
g = BatchNormalization()(g)
g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
g = BatchNormalization()(g)

# feature transformation net
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

# forward net
# g = Lambda(mat_mul, arguments={'B': feature_T})(g)
g = keras.layers.dot(inputs=[g, feature_T], axes=2)
seg_part1 = g
g = Convolution1D(64, 1, activation='relu')(g)
g = BatchNormalization()(g)
g = Convolution1D(128, 1, activation='relu')(g)
g = BatchNormalization()(g)
g = Convolution1D(1024, 1, activation='relu')(g)
g = BatchNormalization()(g)

# global_feature
global_feature = MaxPooling1D(pool_size=num_points)(g)
global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)

# point_net_seg
c = concatenate([seg_part1, global_feature])
c = Convolution1D(512, 1, activation='relu')(c)
c = BatchNormalization()(c)
c = Convolution1D(256, 1, activation='relu')(c)
c = BatchNormalization()(c)
c = Convolution1D(128, 1, activation='relu')(c)
c = BatchNormalization()(c)
c = Convolution1D(128, 1, activation='relu')(c)
c = BatchNormalization()(c)
prediction = Convolution1D(k, 1, activation='softmax')(c)
'''
end of pointnet
'''

# define model
model = Model(inputs=input_points, outputs=prediction)
print(model.summary())

'''
train and evaluate the model
'''
# compile classification model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from DATA.shapenet_generator1 import ShapenetGenerator
train_generator = ShapenetGenerator(directory='./DATA/shapenetcore_partanno_v0', num_points=1024, class_choice='Chair', batch_size=64, train=True)
valid_generator = ShapenetGenerator(directory='./DATA/shapenetcore_partanno_v0', num_points=1024, class_choice='Chair', batch_size=64, train=False)
# from DATA.shapenet_generator2 import ShapenetGenerator
# train_generator = ShapenetGenerator(directory='./DATA/shapenetcore_partanno_segmentation_benchmark_v0', num_points=num_points, cat_choices='Airplane', num_classes=5, batch_size=64, train=True)
# valid_generator = ShapenetGenerator(directory='./DATA/shapenetcore_partanno_segmentation_benchmark_v0', num_points=num_points, cat_choices='Airplane', num_classes=5, batch_size=64, train=False)

# train model
for i in range(epo):
    model.fit_generator(generator=train_generator, validation_data=valid_generator, epochs=2, steps_per_epoch=train_generator.__len__()/8, validation_steps=valid_generator.__len__()/8, shuffle=True, verbose=2)

# evaluate model
score = model.evaluate_generator(generator=valid_generator, verbose=1)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

# save model
model.save('./DATA/saved_model.h5')