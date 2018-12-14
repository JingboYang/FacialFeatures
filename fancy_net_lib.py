import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge,Add, Concatenate
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Deconv2D, UpSampling2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, random, sys, keras
from keras.models import Model

# https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py

LEAK_ALPHA = 0.2
DROPOUT_RATE = 0.25
n_colors = 1

dim_order='tf'

def conv2d(H, n_channel, size=3, stride=1, \
            activation='relu', dropout_rate=0.1, regularization=0.01, \
            normalize=True):

    n_channel = int(n_channel)
    H = Convolution2D(n_channel, size, size, \
                    subsample=(stride, stride), border_mode='same', 
                    dim_ordering=dim_order,\
                    W_regularizer=l2(regularization),\
                    init='glorot_uniform')(H)

    if activation in ['relu', 'tanh', 'sigmoid', 'softmax']:
        H = Activation(activation)(H)
    elif activation == 'leaky':
        H = LeakyReLU(LEAK_ALPHA)(H)
    else:
        print('Warning! No activation')

    if dropout_rate != 0:
        H = Dropout(dropout_rate)(H)

    if normalize:
        H = InstanceNormalization(axis=1)(H)

    return H


def deconv2d(H, prev_input, n_channel, size=3, stride=1, \
                activation='relu', dropout_rate=0.1, regularization=0.05,\
                normalize=True):

    n_channel = int(n_channel)

    #H = UpSampling2D(size=(2, 2))(H)

    H = Convolution2D(n_channel, size, size, \
                    subsample=(stride, stride), border_mode='same', \
                    dim_ordering=dim_order,\
                    W_regularizer=l2(regularization),\
                    init='glorot_uniform')(H)

    if activation in ['relu', 'tanh', 'sigmoid', 'softmax']:
        H = Activation(activation)(H)
    elif activation == 'leaky':
        H = LeakyReLU(LEAK_ALPHA)(H)
    else:
        print('Warning! No activation')

    if dropout_rate != 0:
        H = Dropout(dropout_rate)(H)

    #H = Concatenate(axis=1)([H, prev_input])
    H = Concatenate(axis=3)([H, prev_input])

    if normalize:
        #H = InstanceNormalization(axis=1)(H)
        H = InstanceNormalization(axis=3)(H)

    return H



def res_conv2d(H, n_channel):

    input_vals = H
    H = conv2d(H, n_channel, activation='leaky', dropout_rate=DROPOUT_RATE)
    H = conv2d(H, n_channel, activation='leaky', dropout_rate=DROPOUT_RATE, normalize=False)

    H = Add()([H, input_vals])

    H = InstanceNormalization(axis=1)(H)

    return H


def res_deconv2d(H, prev_input, n_channel):

    input_vals = H
    i1 = UpSampling2D(size=(2, 2))(input_vals)

    H = deconv2d(H, prev_input, n_channel, activation='leaky', dropout_rate=DROPOUT_RATE)
    H = deconv2d(H, H, n_channel, activation='leaky', dropout_rate=DROPOUT_RATE, normalize=False)

    H = Add()([H, i1])

    H = InstanceNormalization(axis=1)(H)

    return H




def generator_default(input_shape, optimizer=None):

    n_filters = 128

    c0 = Input(shape=input_shape)

    #c1 = res_conv2d(c0, n_filters)
    #c2 = res_conv2d(c1, n_filters)
    #c3 = res_conv2d(c2, n_filters)

    c1 = conv2d(c0, n_filters, size=2, activation='leaky', dropout_rate=DROPOUT_RATE)
    c2 = conv2d(c1, n_filters, size=2, activation='leaky', dropout_rate=DROPOUT_RATE)
    c3 = conv2d(c2, n_filters, size=2, activation='leaky', dropout_rate=DROPOUT_RATE)
    c4 = conv2d(c3, n_filters, activation='leaky', dropout_rate=DROPOUT_RATE)
    c5 = conv2d(c4, n_filters, activation='leaky', dropout_rate=DROPOUT_RATE)
    c6 = conv2d(c5, n_filters, activation='leaky', dropout_rate=DROPOUT_RATE)
    c7 = conv2d(c6, n_filters, activation='leaky', dropout_rate=DROPOUT_RATE)

    #d1 = res_deconv2d(c3, c2, n_filters)
    #d2 = res_deconv2d(d1, c1, n_filters)

    #d1 = deconv2d(c4, c3, n_filters, activation='leaky', dropout_rate=DROPOUT_RATE)
    #d2 = deconv2d(d1, c2, n_filters, activation='leaky', dropout_rate=DROPOUT_RATE)
    #d3 = deconv2d(d2, c1, n_filters, activation='leaky', dropout_rate=DROPOUT_RATE)

    #o1 = UpSampling2D(size=(2, 2))(d2)
    o2 = conv2d(c7, n_colors, dropout_rate=0, activation='tanh', normalize=False)

    generator = Model(c0,o2)
    #generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    generator.summary()

    return generator



def discriminator_default(input_shape, optimizer):

    n_filters = 128

    c0 = Input(shape=input_shape)

    #c1 = res_conv2d(c0, n_filters)
    #c2 = res_conv2d(c1, n_filters)
    #c3 = res_conv2d(c2, n_filters)

    c1 = conv2d(c0, n_filters, size=3, activation='leaky', dropout_rate=DROPOUT_RATE)
    c2 = conv2d(c1, n_filters, size=3, activation='leaky', dropout_rate=DROPOUT_RATE)
    c3 = conv2d(c2, n_filters, size=2, activation='leaky', dropout_rate=DROPOUT_RATE)

    o1 = conv2d(c3, n_colors, dropout_rate=0, activation='relu')

    o2 = Flatten()(o1)
    o3 = Dense(256)(o2)

    d_V = Dense(2, activation='softmax')(o3)

    discriminator = Model(c0, d_V)
    discriminator.compile(  loss='categorical_crossentropy', \
                            optimizer=optimizer,\
                            metrics=['accuracy']
                         )
    discriminator.summary() 

    return discriminator




def generator_original(input_shape, optimizer=None):
    """U-Net Generator"""

    gf = 32

    def conv2d(layer_input, filters, f_size=4):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=input_shape)

    # Downsampling
    d1 = conv2d(d0, gf)
    d2 = conv2d(d1, gf*2)
    d3 = conv2d(d2, gf*4)
    d4 = conv2d(d3, gf*8)

    # Upsampling
    u1 = deconv2d(d4, d3, gf*4)
    u2 = deconv2d(u1, d2, gf*2)
    u3 = deconv2d(u2, d1, gf)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(n_colors, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

    generator = Model(d0, output_img)
    generator.summary()
    return generator



def discriminator_original(input_shape, optimizer):

    df = 64

    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    img = Input(shape=input_shape)

    d1 = d_layer(img, df, normalization=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
    o1 = Flatten()(validity)

    d_V = Dense(2, activation='softmax')(o1)

    discriminator = Model(img, d_V)
    discriminator.compile(  loss='categorical_crossentropy', \
                            optimizer=optimizer,\
                            metrics=['accuracy']
                         )
    discriminator.summary() 

    return discriminator