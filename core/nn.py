'''
    Basic building blocks for the model.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys


def conv_layer(data_format, input_tensor, stride=1, padding='VALID', shape=None):
    ''' The standard convolution layer '''
    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)

    kernel = create_conv_kernel(shape)

    if data_format == "NCHW":
        conv_stride = [1, 1, stride, stride]
    else:
        conv_stride = [1, stride, stride, 1]
    conv_out = tf.nn.conv2d(input_tensor, kernel, strides=conv_stride, padding=padding, data_format=data_format)
    relu_out = ReLu_layer(conv_out)

    return relu_out


def fc(input_tensor, units):

    init_op = tf.truncated_normal_initializer(stddev=0.001)
    dense_out = tf.layers.dense(input_tensor, units, kernel_initializer=init_op)

    return  dense_out


def create_conv_kernel(shape=None):
    '''
    :param shape: the shape of kernel to be created
    :return: a tf.tensor
    '''

    init_op = tf.truncated_normal_initializer(stddev=0.001)
    var = tf.get_variable(name='kernel', shape=shape, initializer=init_op)

    return var

def max_pool2d(data_format, input_tensor, stride=2, padding='VALID'):
    ''' The standard max_pool2d with kernel size 2x2 '''

    if data_format == "NCHW":
        pool_size = [1, 1, 2, 2]
        pool_stride = [1, 1, stride, stride]
    else:
        pool_size = [1, 2, 2, 1]
        pool_stride = [1, 2, 2, 1]

    out = tf.nn.max_pool(input_tensor, pool_size, pool_stride, padding, data_format)

    return out

def ReLu_layer(input_tensor):

    relu_out = tf.nn.relu(input_tensor)

    return relu_out
