import tensorflow_addons as tfa
import cv2
import numpy as np
import tensorflow as tf

from tensorflow import keras
from scipy import misc
import keras.backend as K
import tensorflow_addons as tfa
from PIL import Image
from numpy import asarray
import math 




#k=K


from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Dropout, Concatenate, Reshape, BatchNormalization
from tensorflow.python.keras.models import Model, Sequential

from model.common import normalize, denormalize, pixel_shuffle

# This is the model of WDSR with edge extraction


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
    
    x = edge_convert(x, num_filters)

    # main branch
    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    m = BatchNormalization(momentum=0.99, epsilon=0.001)(m)

    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)
    m = conv2d_weightnorm(3 * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
    m = BatchNormalization(momentum=0.99, epsilon=0.001)(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    s = conv2d_weightnorm(3 * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(x)
    s = BatchNormalization(momentum=0.99, epsilon=0.001)(s)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    x = Lambda(denormalize)(x)
    
    model = Model(x_in, x, name="wdsr")
    model.summary()

    return Model(x_in, x, name="wdsr")


def edge_convert(x_in, num_filters):
    ####################################################################################################
    # Sobel Edge Extractor #############################################################################
    ####################################################################################################
    sobel_x = keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_sobelx, strides=(1,1), activation=None)(x_in)
    sobel_y = keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_sobely, strides=(1,1), activation=None)(x_in)
    nsobelx11 = tf.image.convert_image_dtype(sobel_x, dtype=tf.uint8)
    nsobelx12 = tf.cast(nsobelx11, tf.float32)
    nsobely11 = tf.image.convert_image_dtype(sobel_y, dtype=tf.uint8)
    nsobely12 = tf.cast(nsobely11, tf.float32)
    
    ######################################
    # Testing Portion ####################
    ######################################
    #nsobelx1 = sobel_x.numpy()
    #nsobelx2 = nsobelx1.reshape((480,640))
    #nsobelx3 = np.uint8(nsobelx2)
    #nsobely1 = sobel_y.numpy()
    #nsobely2 = nsobely1.reshape((480,640))
    #nsobely3 = np.uint8(nsobely2)
    #new_sobel = tf.image.sobel_edges(tf.convert_to_tensor(x_in))
    #new_sobel = tf.image.sobel_edges(x_in)
    #sobel_x = np.asarray(new_sobel[0, :, :, :, 0])  #  Sobel_X
    #sobel_y = np.asarray(new_sobel[0, :, :, :, 1])  #  Sobel_Y
    #new_sobel_x = np.uint8(sobel_x.reshape((480,640)))
    #new_sobel_y = np.uint8(sobel_y.reshape((480,640)))
    #magx = nsobelx1*nsobelx1
    #magy = nsobely1*nsobely1
    ######################################
    # The End of Testing Portion #########
    ######################################
    
    magx = sobel_x*sobel_x
    magy = sobel_y*sobel_y 
    sq = magx + magy
    #sobel = np.sqrt(sq)
    sq = tf.math.sqrt(sq)
    nsobel11 = tf.image.convert_image_dtype(sq, dtype=tf.uint8)
    sobel = tf.cast(nsobel11, tf.float32)

    ######################################
    # Testing Portion ####################
    ######################################
    #new_sobel1 = np.asarray(sobel)
    #new_sobel2 = new_sobel1.reshape((480,640))
    #new_sobel3 = np.uint8(new_sobel2)
    ######################################
    # The End of Testing Portion #########
    ######################################
    
    ###################################################################################################
    # kirsch Edge Extractor ###########################################################################
    ###################################################################################################
    kirsch1 = keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch1, strides=(1,1), activation=None)(x_in)
    kirsch2 = keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch2, strides=(1,1), activation=None)(x_in)
    kirsch3 = keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch3, strides=(1,1), activation=None)(x_in)
    kirsch4 = keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch4, strides=(1,1), activation=None)(x_in)
    kirsch5 = keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch5, strides=(1,1), activation=None)(x_in)
    kirsch6 = keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch6, strides=(1,1), activation=None)(x_in)
    kirsch7 = keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch7, strides=(1,1), activation=None)(x_in)
    kirsch8 = keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_kirsch8, strides=(1,1), activation=None)(x_in)
    nkirsch11 = tf.image.convert_image_dtype(kirsch1, dtype=tf.uint8)
    nkirsch12 = tf.cast(nkirsch11, tf.float32)
    nkirsch21 = tf.image.convert_image_dtype(kirsch2, dtype=tf.uint8)
    nkirsch22 = tf.cast(nkirsch21, tf.float32)
    nkirsch31 = tf.image.convert_image_dtype(kirsch3, dtype=tf.uint8)
    nkirsch32 = tf.cast(nkirsch31, tf.float32)
    nkirsch41 = tf.image.convert_image_dtype(kirsch4, dtype=tf.uint8)
    nkirsch42 = tf.cast(nkirsch41, tf.float32)
    nkirsch51 = tf.image.convert_image_dtype(kirsch5, dtype=tf.uint8)
    nkirsch52 = tf.cast(nkirsch51, tf.float32)
    nkirsch61 = tf.image.convert_image_dtype(kirsch6, dtype=tf.uint8)
    nkirsch62 = tf.cast(nkirsch61, tf.float32)
    nkirsch71 = tf.image.convert_image_dtype(kirsch7, dtype=tf.uint8)
    nkirsch72 = tf.cast(nkirsch71, tf.float32)
    nkirsch81 = tf.image.convert_image_dtype(kirsch8, dtype=tf.uint8)
    nkirsch82 = tf.cast(nkirsch81, tf.float32)

    ######################################
    # Testing Portion ####################
    ######################################
    #nkirsch11 = kirsch1.numpy()
    #nkirsch12 = nkirsch11.reshape((480,640))
    #nkirsch13 = np.uint8(nkirsch12)
    #nkimg1 = Image.fromarray(nkirsch13, 'L')
    #nkimg1.save('new_kirsch1.png')
    #nkirsch21 = kirsch2.numpy()
    #nkirsch22 = nkirsch21.reshape((480,640))
    #nkirsch23 = np.uint8(nkirsch22)
    #nkimg2 = Image.fromarray(nkirsch23, 'L')
    #nkimg2.save('new_kirsch2.png')
    #nkirsch31 = kirsch3.numpy()
    #nkirsch32 = nkirsch31.reshape((480,640))
    #nkirsch33 = np.uint8(nkirsch32)
    #nkimg3 = Image.fromarray(nkirsch33, 'L')
    #nkimg3.save('new_kirsch3.png')
    #nkirsch41 = kirsch4.numpy()
    #nkirsch42 = nkirsch41.reshape((480,640))
    #nkirsch43 = np.uint8(nkirsch42)
    #nkimg4 = Image.fromarray(nkirsch43, 'L')
    #nkimg4.save('new_kirsch4.png')
    #nkirsch51 = kirsch5.numpy()
    #nkirsch52 = nkirsch51.reshape((480,640))
    #nkirsch53 = np.uint8(nkirsch52)
    #nkimg5 = Image.fromarray(nkirsch53, 'L')
    #nkimg5.save('new_kirsch5.png')
    #nkirsch61 = kirsch6.numpy()
    #nkirsch62 = nkirsch61.reshape((480,640))
    #nkirsch63 = np.uint8(nkirsch62)
    #nkimg6 = Image.fromarray(nkirsch63, 'L')
    #nkimg6.save('new_kirsch6.png')
    #nkirsch71 = kirsch7.numpy()
    #nkirsch72 = nkirsch71.reshape((480,640))
    #nkirsch73 = np.uint8(nkirsch72)
    #nkimg7 = Image.fromarray(nkirsch73, 'L')
    #nkimg7.save('new_kirsch7.png')
    #nkirsch81 = kirsch8.numpy()
    #nkirsch82 = nkirsch81.reshape((480,640))
    #nkirsch83 = np.uint8(nkirsch82)
    #nkimg8 = Image.fromarray(nkirsch83, 'L')
    #nkimg8.save('new_kirsch8.png')
    ######################################
    # The End of Testing Portion #########
    ######################################
    
    max_kirsch1 = tf.math.maximum(nkirsch12, nkirsch22)
    max_kirsch2 = tf.math.maximum(max_kirsch1, nkirsch32)
    max_kirsch3 = tf.math.maximum(max_kirsch2, nkirsch42)
    max_kirsch4 = tf.math.maximum(max_kirsch3, nkirsch52)
    max_kirsch5 = tf.math.maximum(max_kirsch4, nkirsch62)
    max_kirsch6 = tf.math.maximum(max_kirsch5, nkirsch72)
    kirsch = tf.math.maximum(max_kirsch6, nkirsch82)
    
    
    ######################################
    # Testing Portion ####################
    ######################################
    #max_kirsch1 = np.maximum(nkirsch13, nkirsch23)
    #max_kirsch2 = np.maximum(max_kirsch1, nkirsch33)
    #max_kirsch3 = np.maximum(max_kirsch2, nkirsch43)
    #max_kirsch4 = np.maximum(max_kirsch3, nkirsch53)
    #max_kirsch5 = np.maximum(max_kirsch4, nkirsch63)
    #max_kirsch6 = np.maximum(max_kirsch5, nkirsch73)
    #max_kirsch7 = np.maximum(max_kirsch6, nkirsch83)
    #maximg = Image.fromarray(max_kirsch7, 'L')
    #maximg.save('kirsch.png')
    ######################################
    # The End of Testing Portion #########
    ######################################
    
    ##################################################################################################
    # Prewitt Edge Extractor #########################################################################
    ##################################################################################################
    pre_x = keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_prex, strides=(1,1), activation=None)(x_in)
    pre_y = keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', kernel_initializer=kernelInitializer_prey, strides=(1,1), activation=None)(x_in)
    nprex11 = tf.image.convert_image_dtype(pre_x, dtype=tf.uint8)
    nprex12 = tf.cast(nprex11, tf.float32)
    nprey11 = tf.image.convert_image_dtype(pre_y, dtype=tf.uint8)
    nprey12 = tf.cast(nprey11, tf.float32)
    
    ######################################
    # Testing Portion ####################
    ######################################
    #nprex1 = pre_x.numpy()
    #nprex2 = nprex1.reshape((480,640))
    #nprex3 = np.uint8(nprex2)
    #npimgx = Image.fromarray(nprex3, 'L')
    #npimgx.save('new_prex.png')
    #nprey1 = pre_y.numpy()
    #nprey2 = nprey1.reshape((480,640))
    #nprey3 = np.uint8(nprey2)
    #npimgy = Image.fromarray(nprey3, 'L')
    #npimgy.save('new_prey.png')
    ######################################
    # The End of Testing Portion #########
    ######################################
    
    
    magxp = pre_x*pre_x
    magyp = pre_y*pre_y
    sqp = magxp + magyp
    #prewitt = np.sqrt(sqp)
    sqprewitt = tf.math.sqrt(sqp)
    nprewitt11 = tf.image.convert_image_dtype(sqprewitt, dtype=tf.uint8)
    prewitt = tf.cast(nprewitt11, tf.float32)

    ######################################
    # Testing Portion ####################
    ######################################
    #new_prewitt1 = np.asarray(prewitt)
    #new_prewitt2 = new_prewitt1.reshape((480,640))
    #new_prewitt3 = np.uint8(new_prewitt2)
    #npimg = Image.fromarray(new_prewitt3, 'L')
    #npimg.save('new_prewitt.png')
    ######################################
    # The End of Testing Portion #########
    ######################################
    
    #f_sobel = new_sobel3.reshape(x_in.shape)
    #f_kirsch = max_kirsch7.reshape(x_in.shape)
    #f_prewitt = new_prewitt3.reshape(x_in.shape)

    # Concatenate
    #x = tf.concat([x_in, sobel, kirsch, prewitt],2)
    #nx = tf.cast(x_in, tf.uint8)

    x = Concatenate()([x_in, sobel, kirsch, prewitt])

    return x

def kernelInitializer_sobelx(shape, dtype=None):
    #print(shape)    
    sobelx = tf.constant(
        [
            [-1, 0, 1], 
            [-2, 0, 2], 
            [-1, 0, 1]
        ], dtype=dtype )
    #create the missing dims.
    sobelx = tf.reshape(sobelx, (3, 3, 1, 1))

    #print(tf.shape(kirsch))
    #tile the last 2 axis to get the expected dims.
    sobelx = tf.tile(sobelx, (1, 1, shape[-2],shape[-1]))

    #print(tf.shape(kirsch))
    return sobelx

def kernelInitializer_sobely(shape, dtype=None):
    #print(shape)    
    sobely = tf.constant(
        [
            [1, 2, 1], 
            [0, 0, 0], 
            [-1, -2, -1]
        ], dtype=dtype )
    #create the missing dims.
    sobely = tf.reshape(sobely, (3, 3, 1, 1))

    #print(tf.shape(kirsch))
    #tile the last 2 axis to get the expected dims.
    sobely = tf.tile(sobely, (1, 1, shape[-2],shape[-1]))

    #print(tf.shape(kirsch))
    return sobely

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
    x = conv2d_weightnorm(num_filters * expansion, kernel_size, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):

    linear = 0.8
    x1 = conv2d_weightnorm(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
    x1 = BatchNormalization(momentum=0.99, epsilon=0.001)(x1)
    #x2 = Concatenate()([x_in, x1])
    x2 = Dropout(0.5)(x1)

    x3 = conv2d_weightnorm(int(num_filters * linear), 1, padding='same')(x2)
    x3 = BatchNormalization(momentum=0.99, epsilon=0.001)(x3)
    #x4 = Concatenate()([x2, x3])
    x4 = Dropout(0.5)(x3)
    
    x5 = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x4)
    x5 = BatchNormalization(momentum=0.99, epsilon=0.001)(x5)
    #x = Concatenate()([x4, x5])
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x5)
    x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)

    x = Dropout(0.5)(x)

    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    
    x = Add()([x_in, x])
    return x


def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfa.layers.WeightNormalization(Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)


