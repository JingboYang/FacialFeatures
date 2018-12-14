# Standard Python Libraries
import os
import time
import random

# Third Party Modules
import numpy as np
import scipy
import matplotlib.pyplot as plt
from IPython import display
from skimage import exposure

# Custom Modules

try:
    HOME_PATH = os.environ['HOME']
except:
    print('On Windows, using alternative home path')
    HOME_PATH = os.path.expanduser('~\Documents')


DATA_PATH = os.path.join(HOME_PATH, 'FaceDisguiseDatabase')
CROPPED_PATH = os.path.join(DATA_PATH, 'FaceAll_cropped')
TRUTH_PATH = os.path.join(DATA_PATH, 'Ground_Truth')
OUT_PATH = 'output'

TRUTH_LABEL = 'FILE NAME,WIDTH,HEIGHT,SEX,SKIN COLOR,MUSTACHE,BEARD,GLASSES,HAT'
TRUTH_LABEL = TRUTH_LABEL.split(',')
TRUTH_MAP = {}
for i, t in enumerate(TRUTH_LABEL):
    TRUTH_MAP[t] = i

img_size = [64,64]



def plot_figures(figures, nrows = 2, ncols=4):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[ind], cmap=plt.gray())
        #axeslist.ravel()[ind].set_title('{},{}'.format(0,0))
        axeslist.ravel()[ind].set_axis_off()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.05)
    #plt.tight_layout() # optional
    plt.savefig('output/samples.png')
    #plt.show()



def gen_figname(e, start_string, suffix, parent_path='output'):
    string = start_string + '_' + str(e) + '_' + suffix + '.png'
    return os.path.join(parent_path, string)


def plot_training_stats(losses, fname, display=False):

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10,30))

    ax[0].plot(losses["fw_d_l"], label='FW Discriminator Loss', color='b', linestyle='-')
    ax[0].plot(losses["bw_d_l"], label='BW Discriminator Loss', color='b', linestyle='--')
    ax[0].set_title('Discriminator Performance')
    ax[0].set_ylabel('Losses', color='b')
    ax[0].tick_params('y', colors='b')
    ax[0].legend(loc=1)

    ax_0 = ax[0].twinx()
    ax_0.plot(losses["fw_d_a"], label='FW Discriminator Accuracy', color='r', linestyle='-')
    ax_0.plot(losses["bw_d_a"], label='BW Discriminator Accuracy', color='r', linestyle='--')
    ax_0.set_ylabel('Accuracy', color='r')
    ax_0.tick_params('y', colors='r')
    ax_0.legend(loc=9)

    ax[1].plot(losses["g_fw_recon"], label='FW Reconstruction Loss')
    ax[1].plot(losses["g_bw_recon"], label='BW Reconstruction Loss')
    ax[1].plot(losses["g_fw_id"], label='FW Identity Loss')
    ax[1].plot(losses["g_bw_id"], label='BW Identity Loss')
    ax[1].set_title('Generator Performance')
    ax[1].set_ylabel('Losses')
    ax[1].legend()

    avg_dis_loss = (np.array(losses["fw_d_l"]) + np.array(losses["bw_d_l"])) / 2
    avg_reocn_loss = (np.array(losses["g_fw_recon"]) + np.array(losses["g_bw_recon"])) / 2
    avg_id_loss = (np.array(losses["g_fw_id"]) + np.array(losses["g_bw_id"])) / 2
    avg_loss = losses['g_loss']

    ax[2].plot(avg_dis_loss, label='Average Discriminator Loss', color='r')
    ax[2].plot(avg_reocn_loss, label='Average Reconstruction Loss', color='b')
    ax[2].plot(avg_id_loss, label='Average Identity Loss', color='c')
    ax[2].set_title('Combined Model Performance')
    ax[2].set_ylabel('Losses')
    ax[2].tick_params('y')
    ax[2].legend(loc=1)
    ax[2].legend()

    ax_2 = ax[2].twinx()
    ax_2.plot(avg_loss, label='Model Weighted Loss', color='g')
    ax_2.set_ylabel('Losses', color='g')
    ax_2.tick_params('y', colors='g')
    ax_2.legend(loc=9)

    if display:
        plt.show()
    else:
        fig.savefig(fname)
    plt.cla()
    plt.close('all')


def plot_gen(generator_proper, generator_id, XTest, fname):
    #print(XTest_dg.shape)
    count = 4
    figures = XTest[:count]
    generated_images = generator_proper.predict(figures)
    identity_images = generator_id.predict(figures)
    figures = np.concatenate((figures, generated_images, identity_images), axis=0)
    #print('Generating plots. Figure shape: {}'.format(figures.shape))

    fig, axeslist = plt.subplots(ncols=4, nrows=3, figsize=(6.4, 4.8), dpi=100)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[ind,:,:,0], cmap=plt.gray())
        #axeslist.ravel()[ind].set_title('{},{}'.format(0,0))
        axeslist.ravel()[ind].set_axis_off()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.05)

    plt.savefig(fname)
    plt.cla()
    #plt.show()


def preprocess(img):
    img_eq = exposure.equalize_hist(img)
    return img_eq




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
        img = preprocess(img)

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





def build_model_path(model_name, prev):
    start_string = model_name + '_' +  time.strftime('%m_%d__%H_%M_%S', time.localtime())
    result = os.path.join(prev, model_name, start_string)

    return result, start_string


def setup_workspace(p=None):
    
    if p == None:
        p = OUT_PATH
    
    if os.path.exists(p):
        pass
    else:
        os.makedirs(p, exist_ok=True)


def dataset_definition():

    disguise, disguise_label, no_disguise, no_disguise_label = load_facedisguise(True)

    X_train_nd = no_disguise.reshape(no_disguise.shape[0], img_size[0], img_size[1], 1)
    X_train_nd = X_train_nd.astype('float32')
    X_train_nd /= 255

    X_train_dg = disguise.reshape(disguise.shape[0], img_size[0], img_size[1], 1)
    X_train_dg = X_train_dg.astype('float32')
    X_train_dg /= 255

    print('Min and Max in no disguise images {},{}'.format(np.min(X_train_nd), np.max(X_train_nd)))
    print('Min and Max in has disguise images {},{}'.format(np.min(X_train_dg), np.max(X_train_dg)))

    print('X_train shape:', X_train_nd.shape)

    shp = X_train_nd.shape[1:]
    print ('Training data point shape: {}'.format(shp))

    ntrain = int(len(X_train_nd) * 0.75)
    trainidx = random.sample(range(0,X_train_nd.shape[0]), ntrain)
    XT_nd = X_train_nd[trainidx,:,:,:]
    print('XT_nd shape {}'.format(XT_nd.shape))

    testidx = list(set(range(len(no_disguise))) - set(trainidx))
    XTest_nd = X_train_nd[trainidx,:,:,:]
    print('XTest_nd shape {}'.format(XTest_nd.shape))

    ntrain = int(len(X_train_dg) * 0.75)
    trainidx = random.sample(range(0,X_train_dg.shape[0]), ntrain)
    XT_dg = X_train_dg[trainidx,:,:,:]
    print('XT_dg shape {}'.format(XT_dg.shape))

    testidx = list(set(range(len(disguise))) - set(trainidx))
    XTest_dg = X_train_dg[trainidx,:,:,:]
    print('XTest_dg shape {}'.format(XTest_dg.shape))
    return shp, XT_nd, XTest_nd, XT_dg, XTest_dg


def sprint(stuff, level=0):
    
    string = str(stuff)

    if level == 0:
        string = '=' * 8 + '  ' + string + '  ' + '='*8
    elif level == 1:
        string = '\t' + string
    else:
        pass

    return string




if __name__ == '__main__':
    
    losses = {"fw_d_l":[], "fw_d_a":[], \
                    "bw_d_l":[], "bw_d_a":[], \
                    "g_fw_id":[], "g_fw_recon":[], \
                    "g_bw_id":[], "g_bw_recon":[], \
                    "g_loss":[]}

    l = np.array([1,2,3,4,5,6,7,8,10])

    losses["fw_d_l"] = l + 0.1
    losses["fw_d_a"] = -l + 0.2
    losses["bw_d_l"] = l + 0.3
    losses["bw_d_a"] = -l + 0.4
    losses["g_fw_id"] = l + 0.5
    losses["g_bw_id"] = l + 0.6
    losses["g_fw_recon"] = l + 0.7
    losses["g_bw_recon"] = l + 0.8
    losses["g_loss"] = l + 1.0

    plot_training_stats(losses, fname=None, display=True)