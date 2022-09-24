import tensorflow_addons as tfa
import cv2
import numpy as np
import tensorflow as tf
import math
import keras.backend as K
import tensorflow_addons as tfa
import tensorflow_probability as tfp


from tensorflow import keras
from scipy import misc
from statistics import stdev



from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Dropout, Concatenate, Reshape, BatchNormalization, DepthwiseConv2D, MaxPooling2D
from tensorflow.python.keras.models import Model
from model.common import normalize, denormalize, pixel_shuffle



def wdsr_a(scale, num_filters=32, num_res_blocks=8, res_block_expansion=4, res_block_scaling=None):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_a)


def wdsr_b(scale, num_filters=32, num_res_blocks=5, res_block_expansion=6, res_block_scaling=None):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_b)


def wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)
    #########################################################
    #x = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    #########################################################
#     x = Conv2D(1, kernel_size=(3,3), padding='same', kernel_initializer=gaussian_kernel1, strides=(1,1), activation='relu')(x)
    x = edge_convert(x, num_filters)
    
    # main branch
    ##########################################################
    #x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)
    # original input x
    m = conv2d_weightnorm(num_filters, 3, padding='same', activation='relu')(x)
    # mean_org = tf.keras.metrics.Mean()(x_org)
    # sobel input x
#     m_sobel = conv2d_weightnorm(3, 3, padding='same', activation='relu')(x_sobel)
#     # mean_sobel = tf.keras.metrics.Mean()(x_sobel)
#     m_sobel = tf.keras.layers.UpSampling2D(size=(scale, scale), data_format=None, interpolation="nearest")(m_sobel)
#     # kirsch input x
#     m_kirsch = conv2d_weightnorm(3, 3, padding='same', activation='relu')(x_kirsch)
#     # mean_kirsch = tf.keras.metrics.Mean()(x_kirsch)
#     m_kirsch = tf.keras.layers.UpSampling2D(size=(scale, scale), data_format=None, interpolation="nearest")(m_kirsch)
#     # prewitt input x
#     m_prewitt = conv2d_weightnorm(3, 3, padding='same', activation='relu')(x_prewitt)
#     m_prewitt = tf.keras.layers.UpSampling2D(size=(scale, scale), data_format=None, interpolation="nearest")(m_prewitt)
#     # mean_prewitt = tf.keras.metrics.Mean()(x_prewitt)
    ##########################################################
    #m = Conv2D(num_filters, 3, padding='same', strides=(1,1), activation='relu')(x)
    #m = DepthwiseConv2D(3, padding='same', activation='relu')(m)
    #m = Conv2D(num_filters, 1, padding='valid', strides=(1,1), activation='relu')(m)
    #m = Conv2D(num_filters, 3, padding='same', strides=(1,1), activation='relu')(m)
    ##########################################################
    ##########################################################
    # m = BatchNormalization(momentum=0.99, epsilon=0.001)(m)
    ##########################################################

    for i in range(num_res_blocks):
        # orignal input
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)


    # Concatenation
    # m = Concatenate()([m_org, m_sobel, m_kirsch, m_prewitt])
    ###############################################################################################
    #m = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(m)
    m = conv2d_weightnorm(3 * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
    ###############################################################################################
    #m = Conv2D(3 * scale ** 2, 3, padding='same', strides=(1,1), activation='relu')(m)
    #m = DepthwiseConv2D(3, padding='same', activation='relu')(m)
    #m = Conv2D(3 * scale ** 2, 1, padding='valid', strides=(1,1), activation='relu')(m)
    #m = Conv2D(3 * scale ** 2, 3, padding='same', strides=(1,1), activation='relu')(m)
    ###############################################################################################
    ########################################################
    #m = BatchNormalization(momentum=0.99, epsilon=0.001)(m)
    ########################################################
    m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    ###############################################################################################
    s = conv2d_weightnorm(3 * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(x_in)
    ###############################################################################################
    #s = Conv2D(3 * scale ** 2, 5, padding='same', strides=(1,1), activation='relu')(x)
    #s = DepthwiseConv2D(3, padding='same', activation='relu')(s)
    #s = Conv2D(3 * scale ** 2, 1, padding='valid', strides=(1,1), activation='relu')(s)
    #s = Conv2D(3 * scale ** 2, 5, padding='same', strides=(1,1), activation='relu')(s)

    ###############################################################################################
    ########################################################
    #s = BatchNormalization(momentum=0.99, epsilon=0.001)(s)
    ########################################################
    
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
#     x = Add()([x, m_sobel])
#     x = Add()([x, m_kirsch])
#     x = Add()([m, m_prewitt])
    ###########################
    x = Lambda(denormalize)(x)
    ###########################
    
    model = Model(x_in, x, name="wdsr")
    

    return model


################################################################################################


def edge_convert(x_in, num_filters):

    ######################################
    # Sobel Edge Extractor ###############
    ######################################
    ######################################
    # Version 1: #########################
    ######################################
#     sobel_x = keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializerx, strides=(1,1), activation='relu')(x_in)
#     sobel_y = keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializery, strides=(1,1), activation='relu')(x_in)
#     magx = sobel_x*sobel_x
#     magy = sobel_y*sobel_y
#     sq = magx + magy
#     sobel = tf.math.sqrt(sq)
    ######################################
    # kirsch Edge Extractor ##############
    ######################################
    ######################################
    # Version 1: #########################
    ######################################
    kirsch1 = keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch1, strides=(1,1), activation='relu')(x_in)
    kirsch2 = keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch2, strides=(1,1), activation='relu')(x_in)
    kirsch3 = keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch3, strides=(1,1), activation='relu')(x_in)
    kirsch4 = keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch4, strides=(1,1), activation='relu')(x_in)
    kirsch5 = keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch5, strides=(1,1), activation='relu')(x_in)
    kirsch6 = keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch6, strides=(1,1), activation='relu')(x_in)
    kirsch7 = keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch7, strides=(1,1), activation='relu')(x_in)
    kirsch8 = keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch8, strides=(1,1), activation='relu')(x_in)
    kirsch = tf.math.maximum(kirsch1, kirsch2)
    kirsch = tf.math.maximum(kirsch, kirsch3)
    kirsch = tf.math.maximum(kirsch, kirsch4)
    kirsch = tf.math.maximum(kirsch, kirsch5)
    kirsch = tf.math.maximum(kirsch, kirsch6)
    kirsch = tf.math.maximum(kirsch, kirsch7)
    kirsch = tf.math.maximum(kirsch, kirsch8)
    ########################################
    # Prewitt Edge Extractor ###############
    ########################################
    ######################################
    # Version 1: #########################
    ######################################
#     pre_x = keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_prex, strides=(1,1), activation='relu')(x_in)
#     pre_y = keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_prey, strides=(1,1), activation='relu')(x_in)
#     magx = tf.math.multiply(pre_x, pre_x)
#     magy = tf.math.multiply(pre_y, pre_y)
#     sq = tf.math.add(magx, magy)
#     prewitt = tf.math.sqrt(sq)

    ######################################
    # Version 2: #########################
    ######################################
    # Concatenate
#     x = Concatenate()([x_in, sobel, kirsch, prewitt])
    x = Add()([x_in, kirsch])
    x = Concatenate()([x_in, x])

    return x

def gaussian_kernel(shape, dtype=None):
    """Makes 2D gaussian Kernel for convolution."""
    
    size = 1
    
    mean = 0.0
    
    std = 1.0

    d = tfp.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

    gauss_kernel = tf.einsum('i,j->ij',
                                  vals,
                                  vals)
    
    gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
    
    # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
    gauss_kernel = tf.reshape(gauss_kernel, (3, 3, 1, 1))

    #tile the last 2 axis to get the expected dims.
    gauss_kernel = tf.tile(gauss_kernel, (1, 1, shape[-2],shape[-1]))
    
    return gauss_kernel

def kernelInitializerx(shape, dtype=None):
    sobel_x = tf.constant(
        [
            [1, 0, -1], 
            [2, 0, -2], 
            [1, 0, -1]
        ], dtype=dtype )
    #create the missing dims.
    sobel_x = tf.reshape(sobel_x, (3, 3, 1, 1))

    #tile the last 2 axis to get the expected dims.
    sobel_x = tf.tile(sobel_x, (1, 1, shape[-2],shape[-1]))

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
    kirsch = tf.constant(
        [
            [5, 5, 5], 
            [-3, 0, -3], 
            [-3, -3, -3]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    return kirsch

def kernelInitializer_kirsch2(shape, dtype=None):
    kirsch = tf.constant(
        [
            [-3, 5, 5], 
            [-3, 0, 5], 
            [-3, -3, -3]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    return kirsch

def kernelInitializer_kirsch3(shape, dtype=None):
    kirsch = tf.constant(
        [
            [-3, -3, 5], 
            [-3, 0, 5], 
            [-3, -3, 5]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    return kirsch

def kernelInitializer_kirsch4(shape, dtype=None):
    kirsch = tf.constant(
        [
            [-3, -3, -3], 
            [-3, 0, 5], 
            [-3, 5, 5]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    return kirsch

def kernelInitializer_kirsch5(shape, dtype=None):
    kirsch = tf.constant(
        [
            [-3, -3, -3], 
            [-3, 0, -3], 
            [5, 5, 5]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    return kirsch

def kernelInitializer_kirsch6(shape, dtype=None):
    kirsch = tf.constant(
        [
            [-3, -3,-3], 
            [5, 0, -3], 
            [5, 5, -3]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    return kirsch

def kernelInitializer_kirsch7(shape, dtype=None):
    kirsch = tf.constant(
        [
            [5, -3, -3], 
            [5, 0, -3], 
            [5, -3, -3]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    return kirsch

def kernelInitializer_kirsch8(shape, dtype=None):
    kirsch = tf.constant(
        [
            [5, 5, -3], 
            [5, 0, -3], 
            [-3, -3, -3]
        ], dtype=dtype )
    #create the missing dims.
    kirsch = tf.reshape(kirsch, (3, 3, 1, 1))

    #tile the last 2 axis to get the expected dims.
    kirsch = tf.tile(kirsch, (1, 1, shape[-2],shape[-1]))

    return kirsch

def kernelInitializer_prex(shape, dtype=None):
    sobel_x = tf.constant(
        [
            [1, 0, -1], 
            [1, 0, -1], 
            [1, 0, -1]
        ], dtype=dtype )
    #create the missing dims.
    sobel_x = tf.reshape(sobel_x, (3, 3, 1, 1))

    #tile the last 2 axis to get the expected dims.
    sobel_x = tf.tile(sobel_x, (1, 1, shape[-2],shape[-1]))

    return sobel_x

def kernelInitializer_prey(shape, dtype=None):
    sobel_y = tf.constant(
        [
            [1, 1, 1], 
            [0, 0, 0], 
            [-1,-1, -1]
        ], dtype=dtype )
    #create the missing dims.
    sobel_y = tf.reshape(sobel_y, (3, 3, 1, 1))

    #tile the last 2 axis to get the expected dims.
    sobel_y = tf.tile(sobel_y, (1, 1, shape[-2],shape[-1]))

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
    ##########################################################################################
    #x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x_in)
    ##########################################################################################
#     x_max = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="same")(x_in)
#     x_avg = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),strides=(1, 1), padding="same")(x_in)
#     x_max = tf.keras.layers.Dense(num_filters*0.5)(x_max)
#     x_avg = tf.keras.layers.Dense(num_filters*0.5)(x_avg)
#     x_max = tf.keras.layers.LeakyReLU()(x_max)
#     x_avg = tf.keras.layers.LeakyReLU()(x_avg)
#     x_max = tf.keras.layers.Dense(num_filters)(x_max)
#     x_avg = tf.keras.layers.Dense(num_filters)(x_avg)
#     x = Add()([x_max, x_avg])
#     x = tf.keras.activations.sigmoid(x)
#     x = tf.keras.layers.Multiply()([x_in, x])
    ##########################################################################################
    x1 = conv2d_weightnorm(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
    ##########################################################################################
    #x1 = Conv2D(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
    #x1 = DepthwiseConv2D(3, padding='same', activation='relu')(x1)
    #x1 = Conv2D(num_filters * expansion, 1, padding='valid', activation='relu')(x1)
    #x1 = Conv2D(num_filters * expansion, 1, padding='same', activation='relu')(x1)
    ##########################################################
    #x1 = BatchNormalization(momentum=0.99, epsilon=0.001)(x1)
    #x2 = Concatenate()([x_in, x1])
    ##########################################################
    x2 = Dropout(0.5)(x1)

    #########################################################################
    #x3 = conv2d_weightnorm(int(num_filters * linear), 1, padding='same')(x2)
    #x3 = BatchNormalization(momentum=0.99, epsilon=0.001)(x3)
    #x4 = Concatenate()([x2, x3])
    #x4 = Dropout(0.5)(x3)
    #########################################################################
    #x2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x2)
    x5 = conv2d_weightnorm(int(num_filters * linear), 1, padding='same')(x2)
    x5 = Dropout(0.5)(x5)
    x5 = conv2d_weightnorm(num_filters, kernel_size, padding='same', activation='relu')(x5)
    #########################################################################
    #x5 = Conv2D(num_filters, kernel_size, padding='same')(x2)
    #x5 = DepthwiseConv2D(3, padding='same', activation='relu')(x5)
    #x5 = Conv2D(num_filters, 1, padding='valid')(x5)
    #x5 = Conv2D(num_filters, kernel_size, padding='same')(x5)

    #########################################################################
    ############################################################################
    #x5 = BatchNormalization(momentum=0.99, epsilon=0.001)(x5)
    #x = Concatenate()([x4, x5])
    #x = Conv2D(num_filters, kernel_size, padding='same', activation='relu')(x5)
    #x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    ############################################################################
    x = Dropout(0.5)(x5)

    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
        
    ############################################################################
    #x_s = conv2d_weightnorm(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
    #x_s = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x_s)
    ############################################################################
    #x_s = Conv2D(num_filters, 3, padding='same')(x_in)
    #x_s = DepthwiseConv2D(3, padding='same')(x_s)
    #x_s = Conv2D(num_filters, 1, padding='valid')(x_s)
    #x_s = Conv2D(num_filters, 3, padding='same')(x_s)

    
    x = Add()([x_in, x])
    return x

def depth_conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfa.layers.WeightNormalization(DepthwiseConv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)

def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfa.layers.WeightNormalization(Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)

def magnitude_computation(input_x, input_y):
    return np.sqrt(np.square(input_x)+np.square(input_y))

def gaussian_kernel1(shape, dtype=None):
    """Makes 2D gaussian Kernel for convolution."""

    size = 1

    mean = 0.0

    std = 1.0

    d = tfp.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

    gauss_kernel = tf.einsum('i,j->ij',
                                      vals,
                                      vals)

    gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)

    # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
    gauss_kernel = tf.reshape(gauss_kernel, (3, 3, 1, 1))

    #tile the last 2 axis to get the expected dims.
    gauss_kernel = tf.tile(gauss_kernel, (1, 1, shape[-2],shape[-1]))

    return gauss_kernel
