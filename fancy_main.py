# Standard Python Libraries
import os
import sys
import time
import random

# Third Party Modules
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.datasets import mnist

# Custom Modules
from fancy_facegan import cycle_gan_definition, pretrain_discriminator, make_trainable
from utils import plot_training_stats, plot_gen, \
                    gen_figname, plot_figures, \
                    dataset_definition, build_model_path, setup_workspace, \
                    sprint




def train(model_name,\
          data_collection,\
          nets, \
          nb_epoch=5000, plt_frq=25, BATCH_SIZE=32,\
          losses = {"fw_d_l":[], "fw_d_a":[], \
                    "bw_d_l":[], "bw_d_a":[], \
                    "g_fw_id":[], "g_fw_recon":[], \
                    "g_bw_id":[], "g_bw_recon":[], \
                    "g_loss":[]}):

    sprint('Preparing output path', level=1)
    #os.mkdir(build_model_path(model_name, 'output'))
    parent_path, start_string = build_model_path(model_name, 'output')
    setup_workspace(parent_path)

    XT_nd, XTest_nd, XT_dg, XTest_dg = data_collection
    gan, gen_fw, gen_bw, dis_fw, dis_bw = nets

    for e in tqdm(range(nb_epoch)):  
        
        # Select batch
        nd_batch = XT_nd[np.random.randint(0,XT_nd.shape[0],size=BATCH_SIZE),:,:,:]
        dg_batch = XT_dg[np.random.randint(0,XT_dg.shape[0],size=BATCH_SIZE),:,:,:]

        # Generate images
        fw_generated = gen_fw.predict(dg_batch)
        bw_generated = gen_bw.predict(nd_batch)

        # Prepare training 'output'
        fw_X = np.concatenate((nd_batch, fw_generated))
        fw_y = np.zeros([2*BATCH_SIZE,2])
        fw_y[0:BATCH_SIZE,1] = 1
        fw_y[BATCH_SIZE:,0] = 1

        bw_X = np.concatenate((dg_batch, bw_generated))
        bw_y = np.zeros([2*BATCH_SIZE,2])
        bw_y[0:BATCH_SIZE,1] = 1
        bw_y[BATCH_SIZE:,0] = 1

        make_trainable(dis_fw, True)
        fw_d_loss  = dis_fw.train_on_batch(fw_X, fw_y)
        #losses["fw_d_l"].append(fw_d_loss)
        losses["fw_d_l"].append(fw_d_loss[0])
        losses["fw_d_a"].append(fw_d_loss[1])

        make_trainable(dis_bw, True)
        bw_d_loss  = dis_bw.train_on_batch(bw_X, bw_y)
        #losses["bw_d_l"].append(bw_d_loss)
        losses["bw_d_l"].append(bw_d_loss[0])
        losses["bw_d_a"].append(bw_d_loss[1])
    
        # train combined generators
        # Remember that CycleGAN model computes losses as follows
        # Combine Discriminator and Generator
        # gan = Model(inputs=[image_fw, image_bw], \
        #             outputs=[dis_result_fw, dis_result_bw, \
        #                      same_fw,       same_bw,       \
        #                      recovered_fw,  recovered_bw])
        # gan.compile(loss=['binary_crossentropy', 'binary_crossentropy',\
        #                   'mae',                 'mae',                \
        #                   'mae',                 'mae'],\
        #             loss_weights = [1,                1,\
        #                             identity_loss,    identity_loss,\
        #                             consistency_loss, consistency_loss],\
        #             optimizer=optimizer)
        nd_batch_2 = XT_nd[np.random.randint(0,XT_nd.shape[0],size=BATCH_SIZE),:,:,:]
        dg_batch_2 = XT_dg[np.random.randint(0,XT_dg.shape[0],size=BATCH_SIZE),:,:,:]
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1
        
        #make_trainable(dis_fw, False)
        #make_trainable(dis_bw, False)

        g_loss = gan.train_on_batch( [dg_batch_2, nd_batch_2],
                                     [y2,         y2,\
                                      dg_batch_2, nd_batch_2,\
                                      dg_batch_2, nd_batch_2 ]
                                   )
        losses["g_loss"].append(g_loss[0])
        losses["g_fw_id"].append(g_loss[3])
        losses["g_bw_id"].append(g_loss[4])
        losses["g_fw_recon"].append(g_loss[5])
        losses["g_bw_recon"].append(g_loss[6])
        
        # Updates plots
        if e%plt_frq==plt_frq-1:
            #plot_loss(losses, gen_figname(e, start_string, 'loss', parent_path=parent_path))
            plot_training_stats(losses, gen_figname(e, start_string, 'loss', parent_path=parent_path))
            plot_gen(gen_fw, gen_bw, XTest_dg, gen_figname(e, start_string, 'gen_fw', parent_path=parent_path))
            plot_gen(gen_bw, gen_fw, XTest_nd, gen_figname(e, start_string, 'gen_bw', parent_path=parent_path))

    return losses



def main():

    if len(sys.argv) < 2:
        sprint('Not enough arguments, must have model name')
        exit(-1)

    model_name = sys.argv[1]

    sprint('Setting up workspace')
    setup_workspace()

    sprint('Loading dataset')
    input_shape, XT_nd, XTest_nd, XT_dg, XTest_dg = dataset_definition()
    data_collection = (XT_nd, XTest_nd, XT_dg, XTest_dg)

    sprint('Building GAN')
    gan, gen_fw, gen_bw, dis_fw, dis_bw = cycle_gan_definition(input_shape)
    nets = [gan, gen_fw, gen_bw, dis_fw, dis_bw]

    sprint('Pretraining discriminator')
    pretrain_discriminator(XT_nd, XT_dg, gen_fw, gen_bw, dis_fw, dis_bw)

    sprint('Training....')
    train(model_name, \
                data_collection,\
                nets, \
                nb_epoch=1000*100, plt_frq=25,BATCH_SIZE=16)


if __name__ == '__main__':
    main()
