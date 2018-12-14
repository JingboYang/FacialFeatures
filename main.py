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
from facegan import build_discriminator, build_generator, build_stacked_GAN, pretrain_discriminator, make_trainable
from utils import plot_loss, plot_gen, gen_figname, plot_figures


DATA_PATH = os.path.join(os.environ['HOME'], 'FaceDisguiseDatabase')
CROPPED_PATH = os.path.join(DATA_PATH, 'FaceAll_cropped')
TRUTH_PATH = os.path.join(DATA_PATH, 'Ground_Truth')
OUT_APTH = 'output'

img_size = [128, 128]

TRUTH_LABEL = 'FILE NAME,WIDTH,HEIGHT,SEX,SKIN COLOR,MUSTACHE,BEARD,GLASSES,HAT'
TRUTH_LABEL = TRUTH_LABEL.split(',')
TRUTH_MAP = {}
for i, t in enumerate(TRUTH_LABEL):
    TRUTH_MAP[t] = i




def load_truth(truth_file):

    with open(truth_file, 'r') as f:
        lines = f.readlines()
        splits = lines[0].split(',')
        
        #print(splits)
        splits = [splits[s] if s == 0 else splits[s] != '0' for s in range(len(splits))]

        #print(splits)

    return splits


def load_facedisguise(plot=False):
    
    glasses = []
    beard = []

    disguise = []
    disguise_label = []

    no_disguise = []
    no_disguise_label = []

    count = 0
    for fname in os.listdir(CROPPED_PATH):
        
        if False:
            count += 1
            if count > 500:
                break

        full_path = os.path.join(CROPPED_PATH, fname)
        
        img = scipy.ndimage.imread(full_path,mode='L')
        img = scipy.misc.imresize(img, img_size)

        truth_file = os.path.join(TRUTH_PATH, fname).replace('jpg', 'txt').replace(' ', '')
        truth = load_truth(truth_file)

        if truth[TRUTH_MAP['BEARD']] == 1 or truth[TRUTH_MAP['GLASSES']] == 1:
            #disguise.append(img)
            #disguise_label.append([truth[TRUTH_MAP['BEARD']] == 1, truth[TRUTH_MAP['GLASSES']] == 1])

            if truth[TRUTH_MAP['BEARD']] == True and truth[TRUTH_MAP['GLASSES']] == False:
                beard.append(img)
                disguise.append(img)
                disguise_label.append([truth[TRUTH_MAP['BEARD']] == 1, truth[TRUTH_MAP['GLASSES']] == 0])
            elif truth[TRUTH_MAP['BEARD']] == False and truth[TRUTH_MAP['GLASSES']] == True:
                glasses.append(img)
        else:
            no_disguise.append(img)
            no_disguise_label.append([truth[TRUTH_MAP['BEARD']] == 1, truth[TRUTH_MAP['GLASSES']] == 1])
        
        #plt.imshow(img, cmap='gray')
        #plt.show()
        #print(img.shape)
    
    print('Number of imges with no disguse: {}'.format(len(no_disguise)))
    print('Number of imges that have disguse: {}'.format(len(disguise)))

    if plot:
        figures = beard[:4]
        figures.extend(glasses[:4])
        figures.extend(no_disguise[:4])
        plot_figures(figures, 3, 4)

        #plt.imshow(img, cmap='gray')
        #plt.show()
        #print(img.shape)

    return np.array(disguise), np.array(disguise_label), np.array(no_disguise), np.array(no_disguise_label)


def dataset_definition():

    disguise, disguise_label, no_disguise, no_disguise_label = load_facedisguise(True)

    X_train_nd = no_disguise.reshape(no_disguise.shape[0], 1, img_size[0], img_size[1])
    X_train_nd = X_train_nd.astype('float32')
    X_train_nd /= 255

    X_train_dg = disguise.reshape(disguise.shape[0], 1, img_size[0], img_size[1])
    X_train_dg = X_train_dg.astype('float32')
    X_train_dg /= 255

    print('Min and Max in no disguise images {},{}'.format(np.min(X_train_nd), np.max(X_train_nd)))
    print('Min and Max in has disguise images {},{}'.format(np.min(X_train_dg), np.max(X_train_dg)))

    print('X_train shape:', X_train_nd.shape)

    shp = X_train_nd.shape[1:]
    print (shp)

    ntrain = int(len(X_train_nd) * 0.75)
    trainidx = random.sample(range(0,X_train_nd.shape[0]), ntrain)
    XT_nd = X_train_nd[trainidx,:,:,:]
    print('XT_nd shape {}'.format(XT_nd.shape))

    ntrain = int(len(X_train_dg) * 0.75)
    trainidx = random.sample(range(0,X_train_dg.shape[0]), ntrain)
    XT_dg = X_train_dg[trainidx,:,:,:]
    print('XT_dg shape {}'.format(XT_dg.shape))

    testidx = list(set(range(len(disguise))) - set(trainidx))
    XTest_dg = X_train_dg[trainidx,:,:,:]
    print('XTest_dg shape {}'.format(XTest_dg.shape))
    return shp, XT_nd, XT_dg, XTest_dg


def train_for_n(model_name,\
                XT_nd, XT_dg, XTest_dg,\
                generator, discriminator, GAN, \
                nb_epoch=5000, plt_frq=25, BATCH_SIZE=32,\
                losses = {"d":[], "g":[]}):
    
    start_string = model_name + '_' +  time.strftime('%m_%d__%H_%M_%S', time.localtime())
    os.mkdir(os.path.join('output', start_string))
    for e in tqdm(range(nb_epoch)):  
        
        # Make generative images
        image_batch = XT_nd[np.random.randint(0,XT_nd.shape[0],size=BATCH_SIZE),:,:,:]
        disguised_batch = XT_dg[np.random.randint(0,XT_dg.shape[0],size=BATCH_SIZE),:,:,:]
        #print(disguised_batch.shape)
        generated_images = generator.predict(disguised_batch)
        
        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        #print('Batch X shape is {}'.format(X.shape))
        make_trainable(discriminator,True)
        d_loss  = discriminator.train_on_batch(X,y)
        losses["d"].append(d_loss)
    
        # train Generator-Discriminator stack on 
        # input noise to non-generated output class
        non_generated = XT_dg[np.random.randint(0,XT_dg.shape[0],size=BATCH_SIZE),:,:,:]
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1
        
        make_trainable(discriminator,False)
        g_loss = GAN.train_on_batch(non_generated, y2 )
        losses["g"].append(g_loss)
        
        # Updates plots
        if e%plt_frq==plt_frq-1:
            plot_loss(losses, gen_figname(e, start_string, 'loss',))
            plot_gen(generator, XTest_dg, gen_figname(e, start_string, 'gen'))

    return losses



def setup_workspace():
    if os.path.exists(OUT_APTH):
        pass
    else:
        os.mkdir(OUT_APTH)


def main():

    if len(sys.argv) < 2:
        print ('Not enough arguments')
        exit(-1)

    model_name = sys.argv[1]

    print('Setting up workspace')
    setup_workspace()

    print('Loading dataset')
    input_shape, XT_nd, XT_dg, XTest_dg = dataset_definition()
    
    print('Building neural network')
    print('\tBuilding generator')
    generator = build_generator(input_shape)
    print('\tBuilding discriminator')
    discriminator = build_discriminator(input_shape)

    print('Building stacked GAN')
    gan = build_stacked_GAN(input_shape, generator, discriminator)

    print('Pretraining discriminator')
    pretrain_discriminator(XT_nd, XT_dg, generator, discriminator)

    print('Training....')
    train_for_n(model_name, \
                XT_nd, XT_dg, XTest_dg,\
                generator, discriminator, gan, \
                nb_epoch=750, plt_frq=50,BATCH_SIZE=16)


if __name__ == '__main__':
    main()