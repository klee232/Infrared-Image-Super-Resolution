from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model

from model.common import normalize, denormalize, pixel_shuffle

import tensorflow_addons as tfa
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy import misc
import keras.backend as K
import tensorflow_addons as tfa


from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Dropout, Concatenate, Reshape, BatchNormalization
from tensorflow.python.keras.models import Model

from model.common import normalize, denormalize, pixel_shuffle

def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)
    x = edge_convert(x, num_filters)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")

def edge_convert(x_in, num_filters):
    #nx = keras.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu')(x_in)
    ######################################
    # Sobel Edge Extractor ###############
    ######################################
    sobel_x = keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer=kernelInitializerx)(x_in)
    sobel_y = keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer=kernelInitializery)(x_in)
    magx = tf.math.multiply(sobel_x, sobel_x)
    magy = tf.math.multiply(sobel_y, sobel_y)
    sq = tf.math.add(magx, magy)
    sobel = tf.math.sqrt(sq)
    print("Sobel shape:")
    print(sobel.shape)
    ######################################
    # kirsch Edge Extractor ##############
    ######################################
    kirsch1 = keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer=kernelInitializer_kirsch1)(x_in)
    kirsch2 = keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer=kernelInitializer_kirsch2)(x_in)
    kirsch3 = keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer=kernelInitializer_kirsch3)(x_in)
    kirsch4 = keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer=kernelInitializer_kirsch4)(x_in)
    kirsch5 = keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer=kernelInitializer_kirsch5)(x_in)
    kirsch6 = keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer=kernelInitializer_kirsch6)(x_in)
    kirsch7 = keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer=kernelInitializer_kirsch7)(x_in)
    kirsch8 = keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer=kernelInitializer_kirsch8)(x_in)
    kirsch = tf.math.maximum(kirsch1, kirsch2)
    kirsch = tf.math.maximum(kirsch, kirsch3)
    kirsch = tf.math.maximum(kirsch, kirsch4)
    kirsch = tf.math.maximum(kirsch, kirsch5)
    kirsch = tf.math.maximum(kirsch, kirsch6)
    kirsch = tf.math.maximum(kirsch, kirsch7)
    kirsch = tf.math.maximum(kirsch, kirsch8)
    print("Kirsch shape:")
    print(kirsch.shape)
    ######################################
    # Prewitt Edge Extractor ###############
    ######################################
    pre_x = keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer=kernelInitializer_prex)(x_in)
    pre_y = keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer=kernelInitializer_prey)(x_in)
    magx2 = tf.math.multiply(pre_x, pre_x)
    magy2 = tf.math.multiply(pre_y, pre_y)
    sq = tf.math.add(magx2, magy2)
    prewitt = tf.math.sqrt(sq)
    print("Prewitt shape:")
    print(prewitt.shape)

    # Concatenate
    x = Concatenate()([x_in, sobel, kirsch, prewitt])
    print("x shape:")
    print(x.shape)

    return x

def kernelInitializerx(shape, dtype=None):
    #print(shape)    
    sobel_x = tf.constant(
        [
            [1, 0, -1], 
            [2, 0, -2], 
            [1, 0, -1]
        ], dtype=dtype )
    #create the missing dims.
    sobel_x = tf.reshape(sobel_x, (3, 3, 1, 1))

    #print(tf.shape(sobel_x))
    #tile the last 2 axis to get the expected dims.
    sobel_x = tf.tile(sobel_x, (1, 1, shape[-2],shape[-1]))

    #print(tf.shape(sobel_x))
    return sobel_x

def kernelInitializery(shape, dtype=None):
    #print(shape)    
    sobel_y = tf.constant(
        [
            [1, 2, 1], 
            [0, 0, 0], 
            [-1, -2, -1]
        ], dtype=dtype )
    #create the missing dims.
    sobel_y = tf.reshape(sobel_y, (3, 3, 1, 1))

    #print(tf.shape(sobel_y))
    #tile the last 2 axis to get the expected dims.
    sobel_y = tf.tile(sobel_y, (1, 1, shape[-2],shape[-1]))

    #print(tf.shape(sobel_y))
    return sobel_y

def kernelInitializer_kirsch1(shape, dtype=None):
    #print(shape)    
    kirsch = tf.constant(
        [
            [5, 5, 5], 
            [-3, 0, -3], 
            [-3, -3, -3]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #print(tf.shape(kirsch))
    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    #print(tf.shape(kirsch))
    return kirsch

def kernelInitializer_kirsch2(shape, dtype=None):
    #print(shape)    
    kirsch = tf.constant(
        [
            [-3, 5, 5], 
            [-3, 0, 5], 
            [-3, -3, -3]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #print(tf.shape(kirsch))
    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    #print(tf.shape(kirsch))
    return kirsch

def kernelInitializer_kirsch3(shape, dtype=None):
    #print(shape)    
    kirsch = tf.constant(
        [
            [-3, -3, 5], 
            [-3, 0, 5], 
            [-3, -3, 5]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #print(tf.shape(kirsch))
    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    #print(tf.shape(kirsch))
    return kirsch

def kernelInitializer_kirsch4(shape, dtype=None):
    #print(shape)    
    kirsch = tf.constant(
        [
            [-3, -3, -3], 
            [-3, 0, 5], 
            [-3, 5, 5]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #print(tf.shape(kirsch))
    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    #print(tf.shape(kirsch))
    return kirsch

def kernelInitializer_kirsch5(shape, dtype=None):
    #print(shape)    
    kirsch = tf.constant(
        [
            [-3, -3, -3], 
            [-3, 0, -3], 
            [5, 5, 5]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #print(tf.shape(kirsch))
    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    #print(tf.shape(kirsch))
    return kirsch

def kernelInitializer_kirsch6(shape, dtype=None):
    #print(shape)    
    kirsch = tf.constant(
        [
            [-3, -3,-3], 
            [5, 0, -3], 
            [5, 5, -3]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #print(tf.shape(kirsch))
    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    #print(tf.shape(kirsch))
    return kirsch

def kernelInitializer_kirsch7(shape, dtype=None):
    #print(shape)    
    kirsch = tf.constant(
        [
            [5, -3, -3], 
            [5, 0, -3], 
            [5, -3, -3]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #print(tf.shape(kirsch))
    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    #print(tf.shape(kirsch))
    return kirsch

def kernelInitializer_kirsch8(shape, dtype=None):
    #print(shape)    
    kirsch = tf.constant(
        [
            [5, 5, -3], 
            [5, 0, -3], 
            [-3, -3, -3]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #print(tf.shape(kirsch))
    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    #print(tf.shape(kirsch))
    return kirsch

def kernelInitializer_prex(shape, dtype=None):
    #print(shape)    
    sobel_x = tf.constant(
        [
            [1, 0, -1], 
            [1, 0, -1], 
            [1, 0, -1]
        ], dtype=dtype )
    #create the missing dims.
    sobel_x = tf.reshape(sobel_x, (3, 3, 1, 1))

    #print(tf.shape(sobel_x))
    #tile the last 2 axis to get the expected dims.
    sobel_x = tf.tile(sobel_x, (1, 1, shape[-2],shape[-1]))

    #print(tf.shape(sobel_x))
    return sobel_x

def kernelInitializer_prey(shape, dtype=None):
    #print(shape)    
    sobel_y = tf.constant(
        [
            [1, 1, 1], 
            [0, 0, 0], 
            [-1,-1, -1]
        ], dtype=dtype )
    #create the missing dims.
    sobel_y = tf.reshape(sobel_y, (3, 3, 1, 1))

    #print(tf.shape(sobel_y))
    #tile the last 2 axis to get the expected dims.
    sobel_y = tf.tile(sobel_y, (1, 1, shape[-2],shape[-1]))

    #print(tf.shape(sobel_y))
    return sobel_y

def res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x
