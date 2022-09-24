import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec, Layer, Dense, Conv2D, DepthwiseConv2D
from tensorflow.keras import constraints
from tensorflow.keras import initializers

from bnn_ops import binarize
###Clip
class Clip(constraints.Constraint):
    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)
    
###BIN Conv
class BinaryConv2D(Conv2D):
    def __init__(self, filters, kernel_lr_multiplier='Glorot', 
                 bias_lr_multiplier=None, H=1., **kwargs):
        super(BinaryConv2D, self).__init__(filters, **kwargs)
        self.H = H
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        
        
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1 
        if input_shape[channel_axis] is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
            
        base = self.kernel_size[0] * self.kernel_size[1]
        if self.H == 'Glorot':
            nb_input = int(input_dim * base)
            nb_output = int(self.filters * base)
            self.H = np.float32(np.sqrt(1.5 / (nb_input + nb_output)))
            
        if self.kernel_lr_multiplier == 'Glorot':
            nb_input = int(input_dim * base)
            nb_output = int(self.filters * base)
            self.kernel_lr_multiplier = np.float32(1. / np.sqrt(1.5/ (nb_input + nb_output)))

        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=kernel_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight((self.output_dim,),
                                     initializer=self.bias_initializers,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)

        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        binary_kernel = binarize(self.kernel, H=self.H) 
        outputs = K.conv2d(
            inputs,
            binary_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
    
### transition
class transition_layer(tf.keras.layers.Layer):
    def __init__(self, self_filters, **kwargs):
        super(transition_layer, self).__init__()
        self.model = tf.keras.Sequential()
        self.model.add(BinaryConv2D(filters=self_filters
                                    ,kernel_size=(1,1)
                                    ,use_bias=False))
        self.model.add(BinaryConv2D(self_filters
                                    ,kernel_size=(3,3)
                                    ,padding='same'
                                    ,use_bias=False))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.ReLU())
    def call(self, input_tensor, **kwargs):
        Out = self.model(input_tensor)
        return Out
### improved 
class improved_layer(tf.keras.layers.Layer):
    def __init__(self, self_units, self_filters, **kwargs):
        super(improved_layer, self).__init__()
        self.dense = tf.keras.layers.Dense(self_units)
        
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.ReLU()
        self.conv0 = BinaryConv2D(filters=self_filters
                                  ,kernel_size=(1,1)
                                  ,use_bias=False)
        self.conv1 = BinaryConv2D(self_filters
                                  ,kernel_size=(3,3)
                                  ,padding='same'
                                  ,use_bias=False)
        
        self.split = lambda x :tf.split(x, [(self_units-self_filters), self_filters], axis=-1)
        
        self.conv2 = BinaryConv2D(filters=self_units
                                  ,kernel_size=(3,3)
                                  ,padding='same'
                                  ,use_bias=False)
        
    def call(self, input_tensor, **kwargs):           
        split_tensor = self.dense(input_tensor)
        
        x = self.conv0(split_tensor)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        
        y = self.split(split_tensor)
        
        adder = tf.keras.layers.Add()([x, y[1]])
        x = tf.keras.layers.Concatenate()([y[0], adder])
        
        x = self.conv2(x)
        adder =  tf.keras.layers.Add()([x, split_tensor])
        Out = adder
        
        return Out
