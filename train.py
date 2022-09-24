import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
import time
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import math
import numpy as np
import torch
import torch.nn as nn


from model import evaluate
from model import srgan

from sam import SAM


import cv2

from tensorflow import keras
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Dropout, Concatenate, Reshape, BatchNormalization, DepthwiseConv2D, MaxPooling2D

from PIL import Image




class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./ckpt/edsr'):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()

    @property
    def model(self):
        return self.checkpoint.model
    
    

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):

        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint
        


        self.now = time.perf_counter()
        
        #strategy = tf.distribute.MirroredStrategy()
        
        #with strategy.scope():
        self.restore()

        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()
            #with strategy.scope():
            loss = self.train_step(lr, hr)
            loss_mean(loss)

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                # Compute PSNR on validation dataset
                psnr_value = self.evaluate(valid_dataset)

                duration = time.perf_counter() - self.now
                print(f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')

                if save_best_only and psnr_value <= ckpt.psnr and psnr_value != float('inf') :
                    self.now = time.perf_counter()
                    # skip saving checkpoint, no PSNR improvement
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()

    def MeanGradientError(self, outputs, targets, weight=0):
        
        filter_x = tf.tile(tf.expand_dims(tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = outputs.dtype), axis = -1), [1, 1, outputs.shape[-1]])
        filter_x = tf.tile(tf.expand_dims(filter_x, axis = -1), [1, 1, 1, outputs.shape[-1]])
        filter_y = tf.tile(tf.expand_dims(tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = outputs.dtype), axis = -1), [1, 1, targets.shape[-1]])
        filter_y = tf.tile(tf.expand_dims(filter_y, axis = -1), [1, 1, 1, targets.shape[-1]])
        # output gradient
        output_gradient_x = tf.math.square(tf.nn.conv2d(outputs, filter_x, strides = 1, padding = 'SAME'))
        output_gradient_y = tf.math.square(tf.nn.conv2d(outputs, filter_y, strides = 1, padding = 'SAME'))
        #target gradient
        target_gradient_x = tf.math.square(tf.nn.conv2d(targets, filter_x, strides = 1, padding = 'SAME'))
        target_gradient_y = tf.math.square(tf.nn.conv2d(targets, filter_y, strides = 1, padding = 'SAME'))
        # square
        output_gradients = tf.math.sqrt(tf.math.add(output_gradient_x, output_gradient_y))
        target_gradients = tf.math.sqrt(tf.math.add(target_gradient_x, target_gradient_y))
        # compute mean gradient error
        shape = output_gradients.shape[1:3]
        mge = tf.math.reduce_sum(tf.math.squared_difference(output_gradients, target_gradients) / (shape[0] * shape[1]))
       
        return mge * weight
    def gaussian_kernel1(self, shape, dtype=None):
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
    
    def gaussian_kernel2(self, shape, dtype=None):
        """Makes 2D gaussian Kernel for convolution."""

        size = 2

        mean = 0.0

        std = 1.0

        d = tfp.distributions.Normal(mean, std)

        vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

        gauss_kernel = tf.einsum('i,j->ij',
                                      vals,
                                      vals)

        gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)

        # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
        gauss_kernel = tf.reshape(gauss_kernel, (5, 5, 1, 1))

        #tile the last 2 axis to get the expected dims.
        gauss_kernel = tf.tile(gauss_kernel, (1, 1, shape[-2],shape[-1]))

        return gauss_kernel
    
    def gaussian_kernel3(self, shape, dtype=None):
        """Makes 2D gaussian Kernel for convolution."""

        size = 3

        mean = 0.0

        std = 1.0

        d = tfp.distributions.Normal(mean, std)

        vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

        gauss_kernel = tf.einsum('i,j->ij',
                                      vals,
                                      vals)

        gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)

        # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
        gauss_kernel = tf.reshape(gauss_kernel, (7, 7, 1, 1))

        #tile the last 2 axis to get the expected dims.
        gauss_kernel = tf.tile(gauss_kernel, (1, 1, shape[-2],shape[-1]))

        return gauss_kernel
    
    def sobel_function(self, x, y):
        # perform sobel edge detection
        filter_x = tf.tile(tf.expand_dims(tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = x.dtype), axis = -1), [1, 1, x.shape[-1]])
        filter_x = tf.tile(tf.expand_dims(filter_x, axis = -1), [1, 1, 1, x.shape[-1]])
        filter_y = tf.tile(tf.expand_dims(tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = x.dtype), axis = -1), [1, 1, y.shape[-1]])
        filter_y = tf.tile(tf.expand_dims(filter_y, axis = -1), [1, 1, 1, y.shape[-1]])
        # output gradient
        output_gradient_x = tf.math.square(tf.nn.conv2d(x, filter_x, strides = 1, padding = 'SAME'))
        output_gradient_y = tf.math.square(tf.nn.conv2d(x, filter_y, strides = 1, padding = 'SAME'))
        #target gradient
        target_gradient_x = tf.math.square(tf.nn.conv2d(y, filter_x, strides = 1, padding = 'SAME'))
        target_gradient_y = tf.math.square(tf.nn.conv2d(y, filter_y, strides = 1, padding = 'SAME'))
        # square
        output_gradients = tf.math.sqrt(tf.math.add(output_gradient_x, output_gradient_y))
        target_gradients = tf.math.sqrt(tf.math.add(target_gradient_x, target_gradient_y))
        
        return output_gradients, target_gradients
    
    def kirsch_filter(self, x, y):
        # perform sobel edge detection
        filter_1 = tf.tile(tf.expand_dims(tf.constant([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype = x.dtype), axis = -1), [1, 1, x.shape[-1]])
        filter_1 = tf.tile(tf.expand_dims(filter_1, axis = -1), [1, 1, 1, x.shape[-1]])
        filter_2 = tf.tile(tf.expand_dims(tf.constant([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype = x.dtype), axis = -1), [1, 1, y.shape[-1]])
        filter_2 = tf.tile(tf.expand_dims(filter_2, axis = -1), [1, 1, 1, y.shape[-1]])
        filter_3 = tf.tile(tf.expand_dims(tf.constant([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype = x.dtype), axis = -1), [1, 1, x.shape[-1]])
        filter_3 = tf.tile(tf.expand_dims(filter_3, axis = -1), [1, 1, 1, x.shape[-1]])
        filter_4 = tf.tile(tf.expand_dims(tf.constant([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype = x.dtype), axis = -1), [1, 1, y.shape[-1]])
        filter_4 = tf.tile(tf.expand_dims(filter_4, axis = -1), [1, 1, 1, y.shape[-1]])
        filter_5 = tf.tile(tf.expand_dims(tf.constant([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype = x.dtype), axis = -1), [1, 1, x.shape[-1]])
        filter_5 = tf.tile(tf.expand_dims(filter_5, axis = -1), [1, 1, 1, x.shape[-1]])
        filter_6 = tf.tile(tf.expand_dims(tf.constant([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype = x.dtype), axis = -1), [1, 1, y.shape[-1]])
        filter_6 = tf.tile(tf.expand_dims(filter_6, axis = -1), [1, 1, 1, y.shape[-1]])
        filter_7 = tf.tile(tf.expand_dims(tf.constant([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype = x.dtype), axis = -1), [1, 1, x.shape[-1]])
        filter_7 = tf.tile(tf.expand_dims(filter_7, axis = -1), [1, 1, 1, x.shape[-1]])
        filter_8 = tf.tile(tf.expand_dims(tf.constant([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype = x.dtype), axis = -1), [1, 1, y.shape[-1]])
        filter_8 = tf.tile(tf.expand_dims(filter_8, axis = -1), [1, 1, 1, y.shape[-1]])
        # output gradient
        output_gradient_x = tf.math.maximum(tf.nn.conv2d(x, filter_1, strides = 1, padding = 'SAME'), tf.nn.conv2d(x, filter_2, strides = 1, padding = 'SAME'))
        output_gradient_x = tf.math.maximum(output_gradient_x, tf.nn.conv2d(x, filter_3, strides = 1, padding = 'SAME'))
        output_gradient_x = tf.math.maximum(output_gradient_x, tf.nn.conv2d(x, filter_4, strides = 1, padding = 'SAME'))
        output_gradient_x = tf.math.maximum(output_gradient_x, tf.nn.conv2d(x, filter_5, strides = 1, padding = 'SAME'))
        output_gradient_x = tf.math.maximum(output_gradient_x, tf.nn.conv2d(x, filter_6, strides = 1, padding = 'SAME'))
        output_gradient_x = tf.math.maximum(output_gradient_x, tf.nn.conv2d(x, filter_7, strides = 1, padding = 'SAME'))
        output_gradient_x = tf.math.maximum(output_gradient_x, tf.nn.conv2d(x, filter_8, strides = 1, padding = 'SAME'))

        #target gradient
        target_gradient_x = tf.math.maximum(tf.nn.conv2d(y, filter_1, strides = 1, padding = 'SAME'), tf.nn.conv2d(y, filter_2, strides = 1, padding = 'SAME'))
        target_gradient_x = tf.math.maximum(target_gradient_x, tf.nn.conv2d(y, filter_3, strides = 1, padding = 'SAME'))
        target_gradient_x = tf.math.maximum(target_gradient_x, tf.nn.conv2d(y, filter_4, strides = 1, padding = 'SAME'))
        target_gradient_x = tf.math.maximum(target_gradient_x, tf.nn.conv2d(y, filter_5, strides = 1, padding = 'SAME'))
        target_gradient_x = tf.math.maximum(target_gradient_x, tf.nn.conv2d(y, filter_6, strides = 1, padding = 'SAME'))
        target_gradient_x = tf.math.maximum(target_gradient_x, tf.nn.conv2d(y, filter_7, strides = 1, padding = 'SAME'))
        target_gradient_x = tf.math.maximum(target_gradient_x, tf.nn.conv2d(y, filter_8, strides = 1, padding = 'SAME'))

        
        return output_gradient_x, target_gradient_x

    def prewitt_function(self, x, y):
        # perform sobel edge detection
        filter_x = tf.tile(tf.expand_dims(tf.constant([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype = x.dtype), axis = -1), [1, 1, x.shape[-1]])
        filter_x = tf.tile(tf.expand_dims(filter_x, axis = -1), [1, 1, 1, x.shape[-1]])
        filter_y = tf.tile(tf.expand_dims(tf.constant([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype = x.dtype), axis = -1), [1, 1, y.shape[-1]])
        filter_y = tf.tile(tf.expand_dims(filter_y, axis = -1), [1, 1, 1, y.shape[-1]])
        # output gradient
        output_gradient_x = tf.math.square(tf.nn.conv2d(x, filter_x, strides = 1, padding = 'SAME'))
        output_gradient_y = tf.math.square(tf.nn.conv2d(x, filter_y, strides = 1, padding = 'SAME'))
        #target gradient
        target_gradient_x = tf.math.square(tf.nn.conv2d(y, filter_x, strides = 1, padding = 'SAME'))
        target_gradient_y = tf.math.square(tf.nn.conv2d(y, filter_y, strides = 1, padding = 'SAME'))
        # square
        output_gradients = tf.math.sqrt(tf.math.add(output_gradient_x, output_gradient_y))
        target_gradients = tf.math.sqrt(tf.math.add(target_gradient_x, target_gradient_y))
        
        return output_gradients, target_gradients

        
    def crop(self, ar, crop_width, copy=False, order='K'):
        '''Crop numpy array at the borders by crop_width.
        Source: www.github.com/scikit-image.'''

        ar = np.array(ar, copy=False)
        crops = _validate_lengths(ar, crop_width)
        slices = [slice(a, ar.shape[i] - b) for i, (a, b) in enumerate(crops)]

        if copy:
            cropped = np.array(ar[slices], order=order, copy=True)
        else:
            cropped = ar[slices]
        return cropped
    
    def compute_ssim(self, x, y, win_size=11, sigma=1.5, L=255, K1=0.01, K2=0.03):
        ###############################################################################
        # This is the modified version of ssim (known as gradient ssim from Chen et al)
        ###############################################################################
        # constants to avoid numerical instabilities close to zero
        C1 = (K1 * L)**2
        C2 = (K2 * L)**2
        C3 = C2/2
        
        # perform gaussian distribution
        ux = tfa.image.gaussian_filter2d(x, [win_size, win_size], sigma)
        uy = tfa.image.gaussian_filter2d(y, [win_size, win_size], sigma)
        # build luminance function l
        coe1 = tf.math.multiply(ux, uy)
        coe2 = tf.math.multiply(ux, ux)
        coe3 = tf.math.multiply(uy, uy)
        num = 2*coe1 + C1
        den = coe2 + coe3 + C1
        l = tf.math.divide(num, den)
        
        # perform sobel edge detection
        filter_x = tf.tile(tf.expand_dims(tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = x.dtype), axis = -1), [1, 1, x.shape[-1]])
        filter_x = tf.tile(tf.expand_dims(filter_x, axis = -1), [1, 1, 1, x.shape[-1]])
        filter_y = tf.tile(tf.expand_dims(tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = x.dtype), axis = -1), [1, 1, y.shape[-1]])
        filter_y = tf.tile(tf.expand_dims(filter_y, axis = -1), [1, 1, 1, y.shape[-1]])
        # output gradient
        output_gradient_x = tf.math.square(tf.nn.conv2d(x, filter_x, strides = 1, padding = 'SAME'))
        output_gradient_y = tf.math.square(tf.nn.conv2d(x, filter_y, strides = 1, padding = 'SAME'))
        #target gradient
        target_gradient_x = tf.math.square(tf.nn.conv2d(y, filter_x, strides = 1, padding = 'SAME'))
        target_gradient_y = tf.math.square(tf.nn.conv2d(y, filter_y, strides = 1, padding = 'SAME'))
        # square
        output_gradients = tf.math.sqrt(tf.math.add(output_gradient_x, output_gradient_y))
        target_gradients = tf.math.sqrt(tf.math.add(target_gradient_x, target_gradient_y))
        
        # build contrast function cg
        ux1, var_out = tf.nn.moments(output_gradients, axes=[0, 1, 2])
        uy1, var_tar = tf.nn.moments(target_gradients, axes=[0, 1, 2])
        std_out = tf.math.sqrt(var_out)
        std_tar = tf.math.sqrt(var_tar)
        coe1 = tf.math.multiply(std_out, std_tar)
        coe2 = tf.math.multiply(std_out, std_out)
        coe3 = tf.math.multiply(std_tar, std_tar)
        num = 2*coe1 + C2
        den = coe2 + coe3 + C2
        cg = tf.math.divide(num, den)
        
        # build structure function sg
        cov = tfp.stats.covariance(output_gradients, target_gradients, sample_axis=0, event_axis=None)
        coe1 = tf.math.multiply(std_out, std_tar)
        num = cov + C3
        den = coe1 + C3
        sg = tf.math.divide(num, den)
        
        # compute the terms of eq. 13 of Wang et al
        ssim = (l**5)*(cg**9)*(sg)
        #max_ssim = tf.reduce_max(ssim)
        #min_ssim = tf.reduce_min(ssim)
        #norm_ssim = (ssim-min_ssim)/(max_ssim-min_ssim)
        #print("The value of ssim: ")
        #print(norm_ssim)
        # compute mean SSIM not for the border areas where no full sliding window
        # was applied
        pad = (win_size - 1) // 2
        #mssim = self.crop(ssim, pad).mean()
        mssim = tf.math.reduce_mean(ssim)
        #norm_mssim = tf.math.reduce_mean(norm_ssim)
        #print("The value of mssim: ")
        #print(norm_mssim)

        # return mean SSIM as well as the full map
        return mssim, ssim

    
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            # MAE Error
            sr = self.checkpoint.model(lr, training=True)
            mae = self.loss(hr, sr)
            # Detail Loss
            hr_gau = Conv2D(1, kernel_size=(3,3), padding='same', kernel_initializer=self.gaussian_kernel1, strides=(1,1), activation='relu')(hr)
            sr_feat = Conv2D(32, 1, padding='same', activation='relu')(sr)
            hr_feat = Conv2D(32, 1, padding='same', activation='relu')(hr_gau)
            sr_feat2 = Conv2D(32, 3, padding='same', activation='relu')(sr)
            hr_feat2 = Conv2D(32, 3, padding='same', activation='relu')(hr_gau)
            mae_feat = self.loss(hr_feat, sr_feat)
            mae_feat2 = self.loss(hr_feat2, sr_feat2)
            mae_feat = mae_feat + mae_feat2
            # target edge-enhancement loss
            sobel_sr, sobel_hr = self.sobel_function(sr, hr)
            hr_gau1 = Conv2D(1, kernel_size=(3,3), padding='same', kernel_initializer=self.gaussian_kernel1, strides=(1,1), activation='relu')(sobel_hr)
            hr_gau2 = Conv2D(1, kernel_size=(5,5), padding='same', kernel_initializer=self.gaussian_kernel2, strides=(1,1), activation='relu')(sobel_hr)
            hr_gau3 = Conv2D(1, kernel_size=(7,7), padding='same', kernel_initializer=self.gaussian_kernel3, strides=(1,1), activation='relu')(sobel_hr)
            #hr_gau2 = tf.keras.layers.ZeroPadding2D(padding=1)(hr_gau2)
            #hr_gau3 = tf.keras.layers.ZeroPadding2D(padding=2)(hr_gau3)
            g = tf.add(hr_gau1, hr_gau2)
            g = tf.add(g, hr_gau3)
            mae_tee = mae*g
            mssim, ssim = self.compute_ssim(sr, hr, win_size=11, sigma=1.5, L=255, K1=0.01, K2=0.03)
            loss_value = -mssim + mae + mae_feat + mae_tee
        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))
        return loss_value

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            print("Restoring from",self.checkpoint_manager.latest_checkpoint)
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')


class EdsrTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class WdsrTrainer(Trainer):   
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)
        
   

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class SrganGeneratorTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=1e-4):
        super().__init__(model, loss=MeanSquaredError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=1000000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class SrganTrainer:
    #
    # TODO: model and optimizer checkpoints
    #
    def __init__(self,
                 generator,
                 discriminator,
                 content_loss='VGG54',
                 learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])):

        if content_loss == 'VGG22':
            self.vgg = srgan.vgg_22()
        elif content_loss == 'VGG54':
            self.vgg = srgan.vgg_54()
        else:
            raise ValueError("content_loss must be either 'VGG22' or 'VGG54'")

        self.content_loss = content_loss
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = Adam(learning_rate=learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate)

        self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
        self.mean_squared_error = MeanSquaredError()

    def train(self, train_dataset, steps=200000):
        pls_metric = Mean()
        dls_metric = Mean()
        step = 0

        for lr, hr in train_dataset.take(steps):
            step += 1

            pl, dl = self.train_step(lr, hr)
            pls_metric(pl)
            dls_metric(dl)

            if step % 50 == 0:
                print(f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}')
                pls_metric.reset_states()
                dls_metric.reset_states()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.generator(lr, training=True)

            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)

            con_loss = self._content_loss(hr, sr)
            gen_loss = self._generator_loss(sr_output)
            perc_loss = con_loss + 0.001 * gen_loss
            disc_loss = self._discriminator_loss(hr_output, sr_output)

        gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return perc_loss, disc_loss

    @tf.function
    def _content_loss(self, hr, sr):
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    def _generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss
