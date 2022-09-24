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



def wdsr_a(scale, num_filters=32, num_res_blocks=8, res_block_expansion=4, res_block_scaling=None):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_a)


def wdsr_b(scale, num_filters=32, num_res_blocks=8, res_block_expansion=6, res_block_scaling=None):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_b)


def wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)
    #x = conv2d_weightnorm(num_filters, 3, padding='same')(x)

    x_e = edge_convert(x, num_filters)
    #x = BatchNormalization(momentum=0.99, epsilon=0.001)(x_e)

    
    #tf.image.sobel_edges()
    # main branch
    #print(x[1])
    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    #m = BatchNormalization(momentum=0.99, epsilon=0.001)(m)

    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)
    m = conv2d_weightnorm(3 * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
    #m = BatchNormalization(momentum=0.99, epsilon=0.001)(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    s = conv2d_weightnorm(3 * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(x)
    #s = BatchNormalization(momentum=0.99, epsilon=0.001)(s)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    x = Lambda(denormalize)(x)
    
    model = Model(x_in, x, name="wdsr")
    
    #model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1,epsilon=1e-07), 
                      #loss='binary_crossentropy', metrics=['accuracy'])

    return model


def edge_convert(x_in, num_filters):
    #nx = keras.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu')(x_in)
    ######################################
    # Sobel Edge Extractor ###############
    ######################################
    sobel_x = keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializerx, strides=(1,1), activation='relu')(x_in)
    sobel_y = keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializery, strides=(1,1), activation='relu')(x_in)
    magx = tf.math.multiply(sobel_x, sobel_x)
    magy = tf.math.multiply(sobel_y, sobel_y)
    sq = tf.math.add(magx, magy)
    sobel = tf.math.sqrt(sq)
    ######################################
    # kirsch Edge Extractor ##############
    ######################################
    kirsch1 = keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch1, strides=(1,1), activation='relu')(x_in)
    kirsch2 = keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch2, strides=(1,1), activation='relu')(x_in)
    kirsch3 = keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch3, strides=(1,1), activation='relu')(x_in)
    kirsch4 = keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch4, strides=(1,1), activation='relu')(x_in)
    kirsch5 = keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch5, strides=(1,1), activation='relu')(x_in)
    kirsch6 = keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch6, strides=(1,1), activation='relu')(x_in)
    kirsch7 = keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch7, strides=(1,1), activation='relu')(x_in)
    kirsch8 = keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch8, strides=(1,1), activation='relu')(x_in)
    kirsch = tf.math.maximum(kirsch1, kirsch2)
    kirsch = tf.math.maximum(kirsch, kirsch3)
    kirsch = tf.math.maximum(kirsch, kirsch4)
    kirsch = tf.math.maximum(kirsch, kirsch5)
    kirsch = tf.math.maximum(kirsch, kirsch6)
    kirsch = tf.math.maximum(kirsch, kirsch7)
    kirsch = tf.math.maximum(kirsch, kirsch8)
    ######################################
    # Prewitt Edge Extractor ###############
    ######################################
    pre_x = keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_prex, strides=(1,1), activation='relu')(x_in)
    pre_y = keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_prey, strides=(1,1), activation='relu')(x_in)
    magx = tf.math.multiply(pre_x, pre_x)
    magy = tf.math.multiply(pre_y, pre_y)
    sq = tf.math.add(magx, magy)
    prewitt = tf.math.sqrt(sq)

    # Concatenate
    x = Concatenate()([x_in, sobel, kirsch, prewitt])

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


def res_block_a(x_in, num_filters, expansion, kernel_size, scaling):
    x1 = conv2d_weightnorm(num_filters * expansion, kernel_size, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):

    linear = 0.8
    x1 = conv2d_weightnorm(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
    #x1 = BatchNormalization(momentum=0.99, epsilon=0.001)(x1)
    x2 = Concatenate()([x_in, x1])
    x2 = Dropout(0.5)(x2)

    x3 = conv2d_weightnorm(int(num_filters * linear), 1, padding='same')(x2)
    #x3 = BatchNormalization(momentum=0.99, epsilon=0.001)(x3)
    x4 = Concatenate()([x2, x3])
    x4 = Dropout(0.5)(x4)
    
    x5 = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x4)
    #x5 = BatchNormalization(momentum=0.99, epsilon=0.001)(x5)
    x = Concatenate()([x4, x5])
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    #x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)

    x = Dropout(0.5)(x)

    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    
    x = Add()([x_in, x])
    return x


def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfa.layers.WeightNormalization(Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)

def magnitude_computation(input_x, input_y):
    return np.sqrt(np.square(input_x)+np.square(input_y))


