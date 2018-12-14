import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge,Add
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Deconv2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, random, sys, keras
from keras.models import Model

dropout_rate = 0.2

# Multilayer Perceptron Only
def generator_dense(input_shape, opt):
    # Build Generative model ...
    g_input = Input(shape=input_shape)
    H = Flatten()(g_input)

    H = Dense(1024)(H)
    H = BatchNormalization()(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    H = Dense(1024)(H)
    H = BatchNormalization()(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    H = Dense(1024)(H)
    H = BatchNormalization()(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    H = Dense(512)(H)
    H = BatchNormalization()(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    H = Dense(np.prod(input_shape))(H)
    H = BatchNormalization()(H)

    H = Reshape( input_shape )(H)
    
    g_V = Activation('sigmoid')(H)

    generator = Model(g_input,g_V)
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    generator.summary()

    return generator


def discriminator_dense(input_shape, dopt):
    d_input = Input(shape=input_shape)
    H = Flatten()(d_input)

    H = Dense(512)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    H = Dense(512)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    H = Dense(512)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    d_V = Dense(2,activation='softmax')(H)

    discriminator = Model(d_input,d_V)
    discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
    discriminator.summary()

    return discriminator    

#################### Multilayer Perceptron Only ####################


# Simple 6-3 CNN
def generator_conv6(nch, input_shape, opt):
    # Build Generative model ...
    g_input = Input(shape=input_shape)

    H = Convolution2D(int(nch), 3, 3, border_mode='same', init='glorot_uniform')(g_input)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)

    H = Convolution2D(int(nch), 3, 3, border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Dropout(dropout_rate)(H)

    H = Convolution2D(int(nch), 3, 3, border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)

    H = Convolution2D(int(nch), 3, 3, border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Dropout(dropout_rate)(H)

    H = Convolution2D(int(nch), 3, 3, border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)

    H = Convolution2D(int(nch/2), 3, 3, border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)

    H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)

    g_V = Activation('sigmoid')(H)

    generator = Model(g_input,g_V)
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    generator.summary()

    return generator 


def discriminator_conv3(input_shape, dopt):
    d_input = Input(shape=input_shape)

    H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    H = Convolution2D(128, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)

    H = Flatten()(H)

    H = Dense(256)(H)
    H = LeakyReLU(0.2)(H)

    d_V = Dense(2,activation='softmax')(H)

    discriminator = Model(d_input,d_V)
    discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
    discriminator.summary()

    return discriminator

#################### Simple CNN ####################





# Residual CNN
def generator_conv_res_9(nch, input_shape, opt):
    # Build Generative model ...
    g_input = Input(shape=input_shape)

    H = Convolution2D(int(nch), 3, 3, border_mode='same', init='glorot_uniform')(g_input)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Dropout(dropout_rate)(H)
    
    H = Convolution2D(int(nch), 3, 3, border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    
    H = Convolution2D(int(nch), 3, 3, border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H_new = BatchNormalization()(H)
    #H = Activation('relu')(H)
    shortcut = BatchNormalization()(H)
    H = H_new

    H = Convolution2D(int(nch), 3, 3, dilation_rate=(2, 2), border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Dropout(dropout_rate)(H)

    H = Convolution2D(int(nch), 3, 3, dilation_rate=(2, 2), border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H_new = BatchNormalization()(H)
    #H = Activation('relu')(H)
    H_new = Add()([H_new, shortcut])
    shortcut = BatchNormalization()(H_new)
    H = H_new

    H = Convolution2D(int(nch), 3, 3, dilation_rate=(4, 4), border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Dropout(dropout_rate)(H)

    H = Convolution2D(int(nch), 3, 3, dilation_rate=(4, 4), border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H_new = BatchNormalization()(H)
    #H = Activation('relu')(H)
    H_new = Add()([H_new, shortcut])
    shortcut = BatchNormalization()(H_new)
    H = H_new

    H = Convolution2D(int(nch), 3, 3, dilation_rate=(2, 2), border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H = Add()([H, shortcut])
    H = BatchNormalization()(H)
    H = Activation('relu')(H)

    H = Convolution2D(int(nch/2), 3, 3, border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('tanh')(H)

    H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)

    g_V = Activation('sigmoid')(H)

    generator = Model(g_input,g_V)
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    generator.summary()

    return generator


def discriminator_conv5(input_shape, dopt):
    d_input = Input(shape=input_shape)

    H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    H = Convolution2D(128, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    H = Convolution2D(128, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)

    H = Flatten()(H)

    H = Dense(256)(H)
    H = LeakyReLU(0.2)(H)

    d_V = Dense(2,activation='softmax')(H)

    discriminator = Model(d_input,d_V)
    discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
    discriminator.summary()

    return discriminator




# Other Stuff .... 


def generator_original():
    # Build Generative model ...
    nch = 200
    g_input = Input(shape=[100])
    H = Dense(nch*14*14, init='glorot_normal')(g_input)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Reshape( [nch, 14, 14] )(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(int(nch/2), 3, 3, border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Convolution2D(int(nch/4), 3, 3, border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
    g_V = Activation('sigmoid')(H)
    generator = Model(g_input,g_V)
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    generator.summary()

    return generator


def generator_conv3(nch, input_shape, opt):
    # Build Generative model ...
    g_input = Input(shape=input_shape)
    H = Convolution2D(int(nch/2), 3, 3, border_mode='same', init='glorot_uniform')(g_input)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Convolution2D(int(nch/4), 3, 3, border_mode='same', init='glorot_uniform')(H)
    #H = BatchNormalization(mode=2)(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
    g_V = Activation('sigmoid')(H)
    generator = Model(g_input,g_V)
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    generator.summary()

    return generator

    