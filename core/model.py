from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import nn
import sys
import os

class model():
    def __init__(self, params):

        self._data_format = params.get('data_format', None)
        self._batch = params.get('batch', 32)
        self._l2_weight = params.get('l2_weight', 0.0002)
        self._init_lr = params.get('init_lr', 1e-3)
        self._margin = params.get('margin', 0.01)

        if self._data_format is not "NCHW" and self._data_format is not "NHWC":
            sys.exit("Invalid data format. Must be either 'NCHW' or 'NHWC'.")

    def _build_model(self, triplets):
        '''
            :param triplets: batches of triplets, [3*N, 64, 64, 3]
            :return: a model dict containing all Tensors
        '''

        model = {}
        if self._data_format == "NCHW":
            images = tf.transpose(triplets, [0, 3, 1, 2])

        shape_dict = {}
        shape_dict['conv1'] = [8, 8, 3, 16]
        with tf.variable_scope('conv1'):
            model['conv1'] = nn.conv_layer(self._data_format, triplets, 1, 'VALID', shape_dict['conv1']) # [3N,57,57,16]
        model['pool1'] = nn.max_pool2d(self._data_format, model['conv1'], 2, 'VALID') # outsize [3N, 28, 28, 16]

        shape_dict['conv2'] = [5, 5, 16, 7]
        with tf.variable_scope('conv2'):
            model['conv2'] = nn.conv_layer(self._data_format, model['pool1'], 1, 'VALID', shape_dict['conv2']) # [3N,24,24,7]
        model['pool2'] = nn.max_pool2d(self._data_format, model['conv2'], 2, 'SAME') # [3N, 12, 12, 7]

        shape_dict['fc1'] = 256
        with tf.variable_scope('fc1'):
            model['fc1'] = nn.fc(model['pool2'], shape_dict['fc1'])  # [3N, 256]

        shape_dict['fc2'] = 16
        with tf.variable_scope('fc2'):
            model['fc2'] = nn.fc(model['fc1'], shape_dict['fc2'])  # [3N, 16]

        return model

    def _loss(self, input_tensor):
        '''
        :param input_tensor: [3N, 16], tf.float32
                [0:N, :] anchors, [N:2N, :] positives, [2N:3N, :] negatives
        :return: sum of triplet and pair losses, scalar
        '''

        # triplet loss
        diff_pos = input_tensor[0:self._batch, :] - input_tensor[self._batch:2*self._batch, :]  # [N, 16]
        diff_neg = input_tensor[0:self._batch, :] - input_tensor[2*self._batch: 3*self._batch, :]  # [N, 16]
        norm_pos_s = tf.square(tf.norm(diff_pos, ord=2, axis=-1))  # [N,]
        norm_neg_s = tf.square(tf.norm(diff_neg, ord=2, axis=-1))  # [N,]
        loss_trip = tf.reduce_sum(tf.maximum(0.0, 1.0 - tf.div(norm_neg_s, (norm_pos_s + self._margin)))) # scalar

        # pair loss
        loss_pair = tf.reduce_sum(norm_pos_s)

        return loss_trip + loss_pair

    def _l2_loss(self):

        l2_losses = []
        for var in tf.trainable_variables():
            l2_losses.append(tf.nn.l2_loss(var))

        return tf.multiply(self._l2_weight, tf.add_n(l2_losses))

    def train(self, triplets):

        # build model
        model = self._build_model(triplets)
        net_out = model['fc2']  # [3N, 16]
        total_loss = self._loss(net_out) + self._l2_loss()
        tf.summary.scalar('total_loss', total_loss)

        # Cannot display current prediction since it's a descriptor vector
        # optimizer
        train_step = tf.train.AdamOptimizer(self._init_lr).minimize(total_loss)
        print("Model built.")

        return total_loss, train_step