# Standard Python Libraries
import os
import sys
import time
import random

# Third Party Modules
import numpy as np
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model

# Custom Modules
import fancy_net_lib 
from utils import sprint


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def build_generator(generator_name, gen_input):

    sprint('Building generator described in {}'.format(generator_name))
    generator = getattr(fancy_net_lib, generator_name)(gen_input)
    return generator



def build_discriminator(discriminator_name, dis_input,opt):

    sprint('Building discriminator described in {}'.format(discriminator_name))
    discriminator = getattr(fancy_net_lib, discriminator_name)(dis_input, opt)
    return discriminator



def cycle_gan_definition(input_shape):

    consistency_loss = 10.0
    identity_loss = 0.1 * consistency_loss
    #optimizer = Adam(lr=1e-4, beta_1=0.99)
    optimizer = Adam(lr=1e-4, beta_1=0.5)
    #optimizer = Adam(lr=2.5e-5,beta_1=0.95,beta_2=0.999,epsilon=0.1,decay=0.00001)

    generator_name = 'generator_original'
    discriminator_name = 'discriminator_original'

    gen_fw = build_generator(generator_name, input_shape)
    gen_bw = build_generator(generator_name, input_shape)

    dis_fw = build_discriminator(discriminator_name, input_shape, optimizer)
    dis_bw = build_discriminator(discriminator_name, input_shape, optimizer)

    # Forward and backward input images
    image_fw = Input(shape=input_shape)
    image_bw = Input(shape=input_shape)

    # Generated images
    generated_bw = gen_fw(image_fw)
    generated_fw = gen_bw(image_bw)
    print(generated_fw, generated_bw)

    # Recoverd images
    recovered_fw = gen_bw(generated_bw)
    recovered_bw = gen_fw(generated_fw)

    # No change
    same_bw = gen_fw(image_bw)
    same_fw = gen_bw(image_fw)

    # Discriminator Results
    dis_result_fw = dis_fw(image_fw)
    dis_result_bw = dis_bw(image_bw)


    # Combine Discriminator and Generator
    gan = Model(inputs=[image_fw, image_bw], \
                outputs=[dis_result_fw, dis_result_bw, \
                         same_fw,       same_bw,       \
                         recovered_fw,  recovered_bw])
    gan.compile(loss=['categorical_crossentropy', 'categorical_crossentropy',\
                      'mae',                      'mae',                \
                      'mae',                      'mae'],\
                loss_weights = [1,                1,\
                                identity_loss,    identity_loss,\
                                consistency_loss, consistency_loss],\
                optimizer=optimizer)

    return gan, gen_fw, gen_bw, dis_fw, dis_bw


def pretrain_discriminator(XT_nd, XT_dg, generator_fw, generator_bw, discriminator_fw, discriminator_bw):

    # Forward Network
    generated_images = generator_fw.predict(XT_dg)
    X = np.concatenate((XT_nd, generated_images))
    n1 = XT_nd.shape[0]
    n2 = XT_dg.shape[0]
    y = np.zeros([n1+n2,2])
    y[:n1,1] = 1
    y[n1:,0] = 1

    make_trainable(discriminator_fw,True)
    discriminator_fw.fit(X,y, nb_epoch=1, batch_size=32)
    y_hat = discriminator_fw.predict(X)

    y_hat_idx = np.argmax(y_hat,axis=1)
    y_idx = np.argmax(y,axis=1)
    diff = y_idx-y_hat_idx
    n_tot = y.shape[0]
    n_rig = (diff==0).sum()
    acc = n_rig*100.0/n_tot
    print ("Forward network accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot))


    # Bakcward Network
    generated_images = generator_bw.predict(XT_nd)
    X = np.concatenate((XT_dg, generated_images))
    n1 = XT_dg.shape[0]
    n2 = XT_nd.shape[0]
    y = np.zeros([n1+n2,2])
    y[:n1,1] = 1
    y[n1:,0] = 1

    make_trainable(discriminator_bw,True)
    discriminator_bw.fit(X,y, nb_epoch=1, batch_size=32)
    y_hat = discriminator_bw.predict(X)

    y_hat_idx = np.argmax(y_hat,axis=1)
    y_idx = np.argmax(y,axis=1)
    diff = y_idx-y_hat_idx
    n_tot = y.shape[0]
    n_rig = (diff==0).sum()
    acc = n_rig*100.0/n_tot
    print ("Backward network accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot))


