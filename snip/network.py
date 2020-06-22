import tensorflow.compat.v1 as tf
import sys
tf.disable_v2_behavior()

from functools import reduce
from helpers import static_size


def load_network(
        datasource, arch, num_classes,
        initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
        ):
    networks = {
        'lenet300': lambda: LeNet300(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            ),
        'lenet5': lambda: LeNet5(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap),
        'alexnet-v1': lambda: AlexNet(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, k=1),
        'alexnet-v2': lambda: AlexNet(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, k=2),
        'vgg-c': lambda: VGG(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version='C'),
        'vgg-d': lambda: VGG(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version='D'),
        'vgg-like': lambda: VGG(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version='like'),
        'wrn-16-8': lambda: WRN(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, depth=16, k=8),
        'wrn-16-10': lambda: WRN(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, depth=16, k=10),
        'wrn-22-8': lambda: WRN(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, depth=22, k=8),
        'lstm-s': lambda: LSTM(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, 128,),
        'lstm-b': lambda: LSTM(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, 256,),
        'gru-s': lambda: GRU(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, 128,),
        'gru-b': lambda: GRU(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, 256,),
    }
    return networks[arch]()


def get_initializer(initializer, dtype):
    if initializer == 'zeros':
        return tf.zeros_initializer(dtype=dtype)
    elif initializer == 'vs-h':
        return tf.variance_scaling_initializer(dtype=dtype) # VS-H
    elif initializer == 'vs-x':
        return tf.glorot_normal_initializer(dtype=dtype) # VS-X
    elif initializer == 'rn':
        return tf.random_normal_initializer(dtype=dtype) # RN
    elif initializer == 'tn':
        return tf.truncated_normal_initializer(dtype=dtype) # TN
    else:
        raise NotImplementedError


class LeNet300(object):
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 ):
        self.name = 'lenet300'
        self.input_dims = [28, 28, 1] # height, width, channel
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = self.construct_weights(initializer_w_ap, initializer_b_ap, True, 'ap')
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.variable_scope(scope):
            weights['w1'] = tf.get_variable('w1', [784, 300], **w_params)
            weights['w2'] = tf.get_variable('w2', [300, 100], **w_params)
            weights['w3'] = tf.get_variable('w3', [100, 10], **w_params)
            weights['b1'] = tf.get_variable('b1', [300], **b_params)
            weights['b2'] = tf.get_variable('b2', [100], **b_params)
            weights['b3'] = tf.get_variable('b3', [10], **b_params)
        return weights

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        inputs_flat = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
        fc1 = tf.matmul(inputs_flat, weights['w1']) + weights['b1']
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.matmul(fc1, weights['w2']) + weights['b2']
        fc2 = tf.nn.relu(fc2)
        fc3 = tf.matmul(fc2, weights['w3']) + weights['b3']
        return fc3


class LeNet5(object):
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 ):
        self.name = 'lenet5'
        self.input_dims = [28, 28, 1] # height, width, channel
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = self.construct_weights(initializer_w_ap, initializer_b_ap, True, 'ap')
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.variable_scope(scope):
            weights['w1'] = tf.get_variable('w1', [5, 5, 1, 20], **w_params)
            weights['w2'] = tf.get_variable('w2', [5, 5, 20, 50], **w_params)
            weights['w3'] = tf.get_variable('w3', [800, 500], **w_params)
            weights['w4'] = tf.get_variable('w4', [500, 10], **w_params)
            weights['b1'] = tf.get_variable('b1', [20], **b_params)
            weights['b2'] = tf.get_variable('b2', [50], **b_params)
            weights['b3'] = tf.get_variable('b3', [500], **b_params)
            weights['b4'] = tf.get_variable('b4', [10], **b_params)
        return weights

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        conv1 = tf.nn.conv2d(inputs, weights['w1'], [1, 1, 1, 1], 'VALID') + weights['b1']
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        conv2 = tf.nn.conv2d(pool1, weights['w2'], [1, 1, 1, 1], 'VALID') + weights['b2']
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        flatten = tf.reshape(pool2, [-1, reduce(lambda x, y: x*y, pool2.shape.as_list()[1:])])
        fc1 = tf.matmul(flatten, weights['w3']) + weights['b3']
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.matmul(fc1, weights['w4']) + weights['b4'] # logits
        return fc2


class AlexNet(object):
    ''' Similar to Alexnet in terms of the total number of conv and fc layers.

    Conv layers:
        The size of kernels and the number of conv filters are the same as the original.
        Due to the smaller input size (CIFAR rather than IMAGENET) we use different strides.
    FC layers:
        The size of fc layers are controlled by k (multiplied by 1024).
        In the original Alexnet, k=4 making the size of largest fc layers to be 4096.
    '''
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 datasource,
                 num_classes,
                 k,
                 ):
        self.datasource = datasource
        self.num_classes = num_classes
        self.k = k
        self.name = 'alexnet'
        self.input_dims = [64, 64, 3] if self.datasource == 'tiny-imagenet' else [32, 32, 3] # h,w,c
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = self.construct_weights(initializer_w_ap, initializer_b_ap, True, 'ap')
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        k = self.k
        weights = {}
        with tf.variable_scope(scope):
            weights['w1'] = tf.get_variable('w1', [11, 11, 3, 96], **w_params)
            weights['w2'] = tf.get_variable('w2', [5, 5, 96, 256], **w_params)
            weights['w3'] = tf.get_variable('w3', [3, 3, 256, 384], **w_params)
            weights['w4'] = tf.get_variable('w4', [3, 3, 384, 384], **w_params)
            weights['w5'] = tf.get_variable('w5', [3, 3, 384, 256], **w_params)
            weights['w6'] = tf.get_variable('w6', [256, 1024*k], **w_params)
            weights['w7'] = tf.get_variable('w7', [1024*k, 1024*k], **w_params)
            weights['w8'] = tf.get_variable('w8', [1024*k, self.num_classes], **w_params)
            weights['b1'] = tf.get_variable('b1', [96], **b_params)
            weights['b2'] = tf.get_variable('b2', [256], **b_params)
            weights['b3'] = tf.get_variable('b3', [384], **b_params)
            weights['b4'] = tf.get_variable('b4', [384], **b_params)
            weights['b5'] = tf.get_variable('b5', [256], **b_params)
            weights['b6'] = tf.get_variable('b6', [1024*k], **b_params)
            weights['b7'] = tf.get_variable('b7', [1024*k], **b_params)
            weights['b8'] = tf.get_variable('b8', [self.num_classes], **b_params)
        return weights

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        bn_params = {
            'training': is_train,
            'trainable': trainable,
        }
        init_st = 4 if self.datasource == 'tiny-imagenet' else 2
        inputs = tf.nn.conv2d(inputs, weights['w1'], [1,init_st,init_st,1], 'SAME') + weights['b1']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.conv2d(inputs, weights['w2'], [1, 2, 2, 1], 'SAME') + weights['b2']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.conv2d(inputs, weights['w3'], [1, 2, 2, 1], 'SAME') + weights['b3']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.conv2d(inputs, weights['w4'], [1, 2, 2, 1], 'SAME') + weights['b4']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.conv2d(inputs, weights['w5'], [1, 2, 2, 1], 'SAME') + weights['b5']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
        inputs = tf.matmul(inputs, weights['w6']) + weights['b6']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.matmul(inputs, weights['w7']) + weights['b7']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.matmul(inputs, weights['w8']) + weights['b8'] # logits
        return inputs


class VGG(object):
    '''
    Similar to the original VGG.
    Available models:
        - VGG-C
        - VGG-D
        - VGG-like

    Differences:
        The number of parameters in conv layers are the same as the original.
        The number of parameters in fc layers are reduced to 512 (4096 -> 512).
        The number of total parameters are different, not just because of the size of fc layers,
        but also due to the fact that the first fc layer receives 1x1 image rather than 7x7 image
        because the input is CIFAR not IMAGENET.
        No dropout is used. Instead, batch norm is used.

    Other refereneces.
        (1) The original paper:
        - paper: https://arxiv.org/pdf/1409.1556.pdf
        - code: http://www.robots.ox.ac.uk/~vgg/research/very_deep/
        * Dropout between fc layers.
        * There is no BatchNorm.
        (2) VGG-like by Zagoruyko, adapted for CIFAR-10.
        - project and code: http://torch.ch/blog/2015/07/30/cifar.html
        * Differences to the original VGG-16 (1):
            - # of fc layers 3 -> 2, so there are 15 (learnable) layers in total.
            - size of fc layers 4096 -> 512.
            - use BatchNorm and add more Dropout.
    '''
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 datasource,
                 num_classes,
                 version,
                 ):
        self.datasource = datasource
        self.num_classes = num_classes
        self.version = version
        self.name = 'VGG-{}'.format(version)
        self.input_dims = [64, 64, 3] if self.datasource == 'tiny-imagenet' else [32, 32, 3] # h,w,c
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = self.construct_weights(initializer_w_ap, initializer_b_ap, True, 'ap')
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.variable_scope(scope):
            weights['w1'] = tf.get_variable('w1', [3, 3, 3, 64], **w_params)
            weights['w2'] = tf.get_variable('w2', [3, 3, 64, 64], **w_params)
            weights['w3'] = tf.get_variable('w3', [3, 3, 64, 128], **w_params)
            weights['w4'] = tf.get_variable('w4', [3, 3, 128, 128], **w_params)
            weights['b1'] = tf.get_variable('b1', [64], **b_params)
            weights['b2'] = tf.get_variable('b2', [64], **b_params)
            weights['b3'] = tf.get_variable('b3', [128], **b_params)
            weights['b4'] = tf.get_variable('b4', [128], **b_params)
            if self.version == 'C':
                weights['w5'] = tf.get_variable('w5', [3, 3, 128, 256], **w_params)
                weights['w6'] = tf.get_variable('w6', [3, 3, 256, 256], **w_params)
                weights['w7'] = tf.get_variable('w7', [1, 1, 256, 256], **w_params)
                weights['w8'] = tf.get_variable('w8', [3, 3, 256, 512], **w_params)
                weights['w9'] = tf.get_variable('w9', [3, 3, 512, 512], **w_params)
                weights['w10'] = tf.get_variable('w10', [1, 1, 512, 512], **w_params)
                weights['w11'] = tf.get_variable('w11', [3, 3, 512, 512], **w_params)
                weights['w12'] = tf.get_variable('w12', [3, 3, 512, 512], **w_params)
                weights['w13'] = tf.get_variable('w13', [1, 1, 512, 512], **w_params)
                weights['b5'] = tf.get_variable('b5', [256], **b_params)
                weights['b6'] = tf.get_variable('b6', [256], **b_params)
                weights['b7'] = tf.get_variable('b7', [256], **b_params)
                weights['b8'] = tf.get_variable('b8', [512], **b_params)
                weights['b9'] = tf.get_variable('b9', [512], **b_params)
                weights['b10'] = tf.get_variable('b10', [512], **b_params)
                weights['b11'] = tf.get_variable('b11', [512], **b_params)
                weights['b12'] = tf.get_variable('b12', [512], **b_params)
                weights['b13'] = tf.get_variable('b13', [512], **b_params)
            elif self.version == 'D' or self.version == 'like':
                weights['w5'] = tf.get_variable('w5', [3, 3, 128, 256], **w_params)
                weights['w6'] = tf.get_variable('w6', [3, 3, 256, 256], **w_params)
                weights['w7'] = tf.get_variable('w7', [3, 3, 256, 256], **w_params)
                weights['w8'] = tf.get_variable('w8', [3, 3, 256, 512], **w_params)
                weights['w9'] = tf.get_variable('w9', [3, 3, 512, 512], **w_params)
                weights['w10'] = tf.get_variable('w10', [3, 3, 512, 512], **w_params)
                weights['w11'] = tf.get_variable('w11', [3, 3, 512, 512], **w_params)
                weights['w12'] = tf.get_variable('w12', [3, 3, 512, 512], **w_params)
                weights['w13'] = tf.get_variable('w13', [3, 3, 512, 512], **w_params)
                weights['b5'] = tf.get_variable('b5', [256], **b_params)
                weights['b6'] = tf.get_variable('b6', [256], **b_params)
                weights['b7'] = tf.get_variable('b7', [256], **b_params)
                weights['b8'] = tf.get_variable('b8', [512], **b_params)
                weights['b9'] = tf.get_variable('b9', [512], **b_params)
                weights['b10'] = tf.get_variable('b10', [512], **b_params)
                weights['b11'] = tf.get_variable('b11', [512], **b_params)
                weights['b12'] = tf.get_variable('b12', [512], **b_params)
                weights['b13'] = tf.get_variable('b13', [512], **b_params)
            weights['w14'] = tf.get_variable('w14', [512, 512], **w_params)
            weights['b14'] = tf.get_variable('b14', [512], **b_params)
            if not self.version == 'like':
                weights['w15'] = tf.get_variable('w15', [512, 512], **w_params)
                weights['w16'] = tf.get_variable('w16', [512, self.num_classes], **w_params)
                weights['b15'] = tf.get_variable('b15', [512], **b_params)
                weights['b16'] = tf.get_variable('b16', [self.num_classes], **b_params)
            else:
                weights['w15'] = tf.get_variable('w15', [512, self.num_classes], **w_params)
                weights['b15'] = tf.get_variable('b15', [self.num_classes], **b_params)
        return weights

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        def _conv_block(inputs, bn_params, filt, st=1):
            inputs = tf.nn.conv2d(inputs, filt['w'], [1, st, st, 1], 'SAME') + filt['b']
            inputs = tf.layers.batch_normalization(inputs, **bn_params)
            inputs = tf.nn.relu(inputs)
            return inputs

        bn_params = {
            'training': is_train,
            'trainable': trainable,
        }
        init_st = 2 if self.datasource == 'tiny-imagenet' else 1

        inputs = _conv_block(inputs, bn_params, {'w': weights['w1'], 'b': weights['b1']}, init_st)
        inputs = _conv_block(inputs, bn_params, {'w': weights['w2'], 'b': weights['b2']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w3'], 'b': weights['b3']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w4'], 'b': weights['b4']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w5'], 'b': weights['b5']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w6'], 'b': weights['b6']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w7'], 'b': weights['b7']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w8'], 'b': weights['b8']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w9'], 'b': weights['b9']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w10'], 'b': weights['b10']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w11'], 'b': weights['b11']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w12'], 'b': weights['b12']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w13'], 'b': weights['b13']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        assert reduce(lambda x, y: x*y, inputs.shape.as_list()[1:3]) == 1

        inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
        inputs = tf.matmul(inputs, weights['w14']) + weights['b14']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        if not self.version == 'like':
            inputs = tf.matmul(inputs, weights['w15']) + weights['b15']
            inputs = tf.layers.batch_normalization(inputs, **bn_params)
            inputs = tf.nn.relu(inputs)
            inputs = tf.matmul(inputs, weights['w16']) + weights['b16']
        else:
            inputs = tf.matmul(inputs, weights['w15']) + weights['b15']

        return inputs


class WRN(object):
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 datasource,
                 num_classes,
                 depth,
                 k,
                 ):
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        self.num_block = (depth - 4) // 6
        self.widths = [int(v * k) for v in (16, 32, 64)]
        self.datasource = datasource
        self.num_classes = num_classes
        self.name = 'WRN-{}-{}'.format(depth, k)
        self.input_dims = [64, 64, 3] if self.datasource == 'tiny-imagenet' else [32, 32, 3] # h,w,c
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = self.construct_weights(initializer_w_ap, initializer_b_ap, True, 'ap')
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.variable_scope(scope):
            weights['init-w0'] = tf.get_variable('init-w0', [3, 3, 3, 16], **w_params)
            weights['init-b0'] = tf.get_variable('init-b0', [16], **b_params)
            self.get_group_weights_biases(weights, 'group0', 
                                          16, self.widths[0], 
                                          w_params, b_params,
                                          self.num_block)
            self.get_group_weights_biases(weights, 'group1', 
                                          self.widths[0], self.widths[1], 
                                          w_params, b_params,
                                          self.num_block)
            self.get_group_weights_biases(weights, 'group2', 
                                          self.widths[1], self.widths[2], 
                                          w_params, b_params,
                                          self.num_block)
            weights['fc-w0'] = tf.get_variable('fc-w0', 
                                               [self.widths[2], self.num_classes], 
                                               **w_params)
            weights['fc-b0'] = tf.get_variable('fc-b0',
                                               [self.num_classes], 
                                               **b_params)
            # for k in weights:
            #     print(k)
            #     print(weights[k])
            #     print()
            # sys.exit("TODO: construct WRN model")
        return weights
    
    def get_group_weights_biases(self, weights, tag_groub, ni, no, w_params, b_params, count_block):
        for i in range(count_block):
            tag_block = '{}-block{}'.format(tag_groub, i)
            self.get_block_weights_biases(weights, tag_block, 
                                          ni if i == 0 else no, no, 
                                          w_params, b_params)

    def get_block_weights_biases(self, weights, tag_block, ni, no, w_params, b_params):
        tag_w0 = '{}-w0'.format(tag_block)
        weights[tag_w0] = tf.get_variable(tag_w0, [3, 3, ni, no], **w_params)
        tag_b0 = '{}-b0'.format(tag_block)
        weights[tag_b0] = tf.get_variable(tag_b0, [no], **b_params)
        tag_w1 = '{}-w1'.format(tag_block)
        weights[tag_w1] = tf.get_variable(tag_w1, [3, 3, no, no], **w_params)
        tag_b1 = '{}-b1'.format(tag_block)
        weights[tag_b1] = tf.get_variable(tag_b1, [no], **b_params)
        if ni != no:
            tag_w2 = '{}-w2'.format(tag_block)
            weights[tag_w2] = tf.get_variable(tag_w2, [1, 1, ni, no], **w_params)
            tag_b2 = '{}-b2'.format(tag_block)
            weights[tag_b2] = tf.get_variable(tag_b2, [no], **b_params)
        return

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        bn_params = {
            'training': is_train,
            'trainable': trainable,
        }
        init_st = 2 if self.datasource == 'tiny-imagenet' else 1

        x = tf.nn.conv2d(inputs, weights['init-w0'], [1, init_st, init_st, 1], 'SAME') + weights['init-b0']
        # print('x shape', x.shape)
        g0 = self.forward_pass_group(weights, 'group0', x, is_train, trainable, bn_params, 1, self.num_block)
        # print('g0 shape', g0.shape)
        g1 = self.forward_pass_group(weights, 'group1', g0, is_train, trainable, bn_params, 2, self.num_block)
        # print('g1 shape', g1.shape)
        g2 = self.forward_pass_group(weights, 'group2', g1, is_train, trainable, bn_params, 2, self.num_block)
        # print('g1 shape', g2.shape)
        o = tf.layers.batch_normalization(g2, **bn_params)
        o = tf.nn.relu(o)
        # print('o shape', o.shape)
        # global average pooling
        gap = tf.layers.flatten(tf.reduce_mean(o, axis=[1, 2], keepdims=True))
        # print('gap shape', gap.shape)
        logits = tf.matmul(gap, weights['fc-w0']) + weights['fc-b0']
        # print('logits shape', logits.shape)
        return logits

    def forward_pass_group(self, weights, tag_groub, inputs, 
                           is_train, trainable, bn_params, stride, count_block):
        o = inputs
        for i in range(count_block):
            tag_block = '{}-block{}'.format(tag_groub, i)
            o = self.forward_pass_block(weights, tag_block, o,
                                        is_train, trainable, bn_params, 
                                        stride if i == 0 else 1)
        return o
    
    def forward_pass_block(self, weights, tag_block, x, 
                           is_train, trainable, bn_params, stride):
        tag_w0 = '{}-w0'.format(tag_block)
        tag_b0 = '{}-b0'.format(tag_block)
        o1 = self.forward_pass_bn_relu(x, bn_params)
        y = self.forward_pass_conv2d(o1, {'w': weights[tag_w0], 'b': weights[tag_b0]}, stride)

        tag_w1 = '{}-w1'.format(tag_block)
        tag_b1 = '{}-b1'.format(tag_block)
        o2 = self.forward_pass_bn_relu(y, bn_params)
        z = self.forward_pass_conv2d(o2, {'w': weights[tag_w1], 'b': weights[tag_b1]}, 1)
        
        tag_w2 = '{}-w2'.format(tag_block)
        tag_b2 = '{}-b2'.format(tag_block)
        if tag_w2 in weights:
            return z + self.forward_pass_conv2d(o1, {'w': weights[tag_w2], 'b': weights[tag_b2]}, stride)
        else:
            return z + x
    
    def forward_pass_bn_relu(self, inputs, bn_params):
        o = tf.layers.batch_normalization(inputs, **bn_params)
        o = tf.nn.relu(o)
        return o
    
    def forward_pass_conv2d(self, inputs, filt, st=1):
        o = tf.nn.conv2d(inputs, filt['w'], [1, st, st, 1], 'SAME') + filt['b']
        return o


class LSTM(object):
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 datasource,
                 num_classes,
                 hidden_unit
                 ):
        self.datasource = datasource
        self.num_classes = num_classes
        self.name = 'LSTM'
        self.n_chunks = 28
        self.input_nodes = 28
        self.hidden_unit = hidden_unit
        self.input_dims = [28, 28, 1]  # height, width, channel
        self.batch_size = 100
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = self.construct_weights(initializer_w_ap, initializer_b_ap, True, 'ap')
        self.weights_ = None
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        u_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }

        weights = {}
        with tf.variable_scope(scope):
            weights['wi'] = tf.get_variable('wi', [self.input_nodes, self.hidden_unit], **w_params)
            weights['ui'] = tf.get_variable('ui', [self.hidden_unit, self.hidden_unit], **u_params)
            weights['bi'] = tf.get_variable('bi', [self.hidden_unit], **b_params)

            weights['wf'] = tf.get_variable('wf', [self.input_nodes, self.hidden_unit], **w_params)
            weights['uf'] = tf.get_variable('uf', [self.hidden_unit, self.hidden_unit], **u_params)
            weights['bf'] = tf.get_variable('bf', [self.hidden_unit], **b_params)

            weights['wog'] = tf.get_variable('wog', [self.input_nodes, self.hidden_unit], **w_params)
            weights['uog'] = tf.get_variable('uog', [self.hidden_unit, self.hidden_unit], **u_params)
            weights['bog'] = tf.get_variable('bog', [self.hidden_unit], **b_params)

            weights['wc'] = tf.get_variable('wc', [self.input_nodes, self.hidden_unit], **w_params)
            weights['uc'] = tf.get_variable('uc', [self.hidden_unit, self.hidden_unit], **u_params)
            weights['bc'] = tf.get_variable('bc', [self.hidden_unit], **b_params)

            weights['fc-w'] = tf.get_variable('fc-w', [self.hidden_unit, self.num_classes], **w_params)
            weights['fc-b'] = tf.get_variable('fc-b', [self.num_classes], **b_params)

        return weights

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        reshaped_inputs = tf.reshape(inputs, [-1, self.n_chunks, self.input_nodes])

        batch_input_ = tf.transpose(reshaped_inputs, [2, 0, 1])
        processed_input = tf.transpose(batch_input_)

        initial_hidden = reshaped_inputs[:, 0, :]
        initial_hidden = tf.matmul(initial_hidden, tf.zeros([self.input_nodes, self.hidden_unit]))
        initial_hidden = tf.stack([initial_hidden, initial_hidden])

        self.weights_ = weights
        all_hidden_states = self.forward_pass_states(processed_input, initial_hidden)
        all_outputs = tf.map_fn(self.forward_pass_output, all_hidden_states)

        return all_outputs[-1]

    def forward_pass_states(self, processed_input, initial_hidden):
        all_hidden_states = tf.scan(
            self.forward_pass_lstm,
            processed_input,
            initializer=initial_hidden,
            name='states')
        all_hidden_states = all_hidden_states[:, 0, :, :]
        return all_hidden_states

    def forward_pass_lstm(self, previous_hidden_memory_tuple, x):
        previous_hidden_state, c_prev = tf.unstack(previous_hidden_memory_tuple)

        i = tf.sigmoid(tf.matmul(x, self.weights_['wi']) +
                       tf.matmul(previous_hidden_state, self.weights_['ui']) + self.weights_['bi'])

        f = tf.sigmoid(tf.matmul(x, self.weights_['wf']) +
                       tf.matmul(previous_hidden_state, self.weights_['uf']) + self.weights_['bf'])

        o = tf.sigmoid(tf.matmul(x, self.weights_['wog']) +
                       tf.matmul(previous_hidden_state, self.weights_['uog']) + self.weights_['bog'])

        c_ = tf.nn.tanh(tf.matmul(x, self.weights_['wc']) +
                        tf.matmul(previous_hidden_state, self.weights_['uc']) + self.weights_['bc'])

        c = f * c_prev + i * c_
        current_hidden_state = o * tf.nn.tanh(c)

        return tf.stack([current_hidden_state, c])

    def forward_pass_output(self, hidden_state):
        return tf.matmul(hidden_state, self.weights_['fc-w']) + self.weights_['fc-b']


class GRU(object):
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 datasource,
                 num_classes,
                 hidden_unit
                 ):
        self.datasource = datasource
        self.num_classes = num_classes
        self.name = 'GRU'
        self.n_chunks = 28
        self.input_nodes = 28
        self.hidden_unit = hidden_unit
        self.input_dims = [28, 28, 1]  # height, width, channel
        self.batch_size = 100
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = self.construct_weights(initializer_w_ap, initializer_b_ap, True, 'ap')
        self.weights_ = None
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }

        weights = {}
        with tf.variable_scope(scope):
            weights['wr'] = tf.get_variable('wr', [self.input_nodes, self.hidden_unit], **w_params)
            weights['br'] = tf.get_variable('br', [self.hidden_unit], **b_params)

            weights['wz'] = tf.get_variable('wz', [self.input_nodes, self.hidden_unit], **w_params)
            weights['bz'] = tf.get_variable('bz', [self.hidden_unit], **b_params)

            weights['wx'] = tf.get_variable('wx', [self.input_nodes, self.hidden_unit], **w_params)
            weights['wh'] = tf.get_variable('wh', [self.hidden_unit, self.hidden_unit], **w_params)

            weights['wo'] = tf.get_variable('wo', [self.hidden_unit, self.num_classes], **w_params)
            weights['bo'] = tf.get_variable('bo', [self.num_classes], **b_params)

        return weights

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        reshaped_inputs = tf.reshape(inputs, [-1, self.n_chunks, self.input_nodes])

        batch_input_ = tf.transpose(reshaped_inputs, [2, 0, 1])
        processed_input = tf.transpose(batch_input_)

        initial_hidden = reshaped_inputs[:, 0, :]
        initial_hidden = tf.matmul(initial_hidden, tf.zeros([self.input_nodes, self.hidden_unit]))

        self.weights_ = weights
        all_hidden_states = self.forward_pass_states(processed_input, initial_hidden)
        all_outputs = tf.map_fn(self.forward_pass_output, all_hidden_states)

        return all_outputs[-1]

    def forward_pass_states(self, processed_input, initial_hidden):
        all_hidden_states = tf.scan(
            self.forward_pass_gru,
            processed_input,
            initializer=initial_hidden,
            name='states')
        return all_hidden_states

    def forward_pass_gru(self, previous_hidden_state, x):
        z = tf.sigmoid(tf.matmul(x, self.weights_['wz']) + self.weights_['bz'])
        r = tf.sigmoid(tf.matmul(x, self.weights_['wr']) + self.weights_['br'])

        h_ = tf.tanh(tf.matmul(x, self.weights_['wx']) +
                     tf.matmul(previous_hidden_state, self.weights_['wh']) * r)

        current_hidden_state = tf.multiply((1 - z), h_) + tf.multiply(previous_hidden_state, z)

        return current_hidden_state

    def forward_pass_output(self, hidden_state):
        return tf.nn.relu(tf.matmul(hidden_state, self.weights_['wo']) + self.weights_['bo'])