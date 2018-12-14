# Standard Python Libraries
import os
import sys
import random
import pickle
os.environ["KERAS_BACKEND"] = "tensorflow"
sys.path.append("../common")

# Third Party Modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import theano as th
import theano.tensor as T
import keras
import keras.models as models
from keras.utils import np_utils
from keras.layers import Input,merge
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
from keras.models import Model
from keras.utils import np_utils
from tqdm import tqdm
from IPython import display

# Custom Modules
from net_library import generator_original, generator_conv3, generator_conv6, generator_dense, generator_conv_res_9
from net_library import discriminator_conv3, discriminator_conv5, discriminator_dense


K.set_image_dim_ordering('th')


dropout_rate = 0.25

opt = Adam(lr=1e-3)
dopt = Adam(lr=1e-4)
#opt = Adam(lr=1e-3)
#opt = Adamax(lr=1e-4)
#opt = Adam(lr=0.0002)
#opt = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
nch = 200

def make_trainable(net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val


def build_generator(input_shape, nch=128):
    #return generator_conv3(nch, input_shape, opt)
    #return generator_conv6(nch, input_shape, opt)
    return generator_conv_res_9(nch, input_shape, opt)
    #return generator_dense(input_shape, opt)


def build_discriminator(input_shape):
    #return discriminator_conv3(input_shape, dopt)
    return discriminator_conv5(input_shape, dopt)
    #return discriminator_dense(input_shape, dopt)


def build_stacked_GAN(input_shape, generator, discriminator):

    # Freeze weights in the discriminator for stacked training
    
    make_trainable(discriminator, False)
    # Build stacked GAN model
    gan_input = Input(shape=input_shape)
    H = generator(gan_input)
    gan_V = discriminator(H)
    GAN = Model(gan_input, gan_V)
    GAN.compile(loss='categorical_crossentropy', optimizer=opt)
    GAN.summary()

    return GAN


def pretrain_discriminator(XT_nd, XT_dg, generator, discriminator):

    # Pre-train the discriminator network ...
    #print(XT_dg.shape)
    generated_images = generator.predict(XT_dg)
    X = np.concatenate((XT_nd, generated_images))
    n1 = XT_nd.shape[0]
    n2 = XT_dg.shape[0]
    y = np.zeros([n1+n2,2])
    y[:n1,1] = 1
    y[n1:,0] = 1

    make_trainable(discriminator,True)
    discriminator.fit(X,y, nb_epoch=1, batch_size=32)
    y_hat = discriminator.predict(X)

    y_hat_idx = np.argmax(y_hat,axis=1)
    y_idx = np.argmax(y,axis=1)
    diff = y_idx-y_hat_idx
    n_tot = y.shape[0]
    n_rig = (diff==0).sum()
    acc = n_rig*100.0/n_tot
    print ("Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot))

