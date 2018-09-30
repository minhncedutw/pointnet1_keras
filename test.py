import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
from keras import optimizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Reshape
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda, concatenate

def mat_mul(A, B):
    return tf.matmul(A, B)

num_points = 5

input_points = Input(shape=(num_points, 3))
x = Convolution1D(16, 1, activation='relu', input_shape=(num_points, 3))(input_points)
y = Dense(9, weights=[np.zeros([16, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
z = Dense(9, weights=[np.zeros([3, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(input_points)
# input_T = Reshape((3, 3))(x)
# z = keras.layers.dot(inputs=[input_points, input_T], axes=2)
model = Model(inputs=input_points, outputs=[x, y, z])

print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

x=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]])
a = model.predict(x=x, batch_size=1)
print(x.shape)
print(a[0].shape)
print(a[1].shape)
print(a[2].shape)
