#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from CNN_Model import DeepSTORM, project_01, normalize_im
from skimage import io
from scipy.io import loadmat, savemat
from os.path import abspath

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_model(datafile, weights_file, meanstd_file, savename, \
        upsampling_factor=8, debug=0):

    # load the tiff image data
    Images = io.imread(datafile)
    (K, M, N) = Images.shape

    # images must be upsampled for input to NN
    Images_upsampled = np.zeros([K, M*upsampling_factor, N*upsampling_factor])
    for i in range(K):
        Images_upsampled[i, ...] = np.kron(Images[i, ...], \
                np.ones([upsampling_factor, upsampling_factor]))
    Images = Images_upsampled.astype('float32')

    # load DeepSTORM network
    model = DeepSTORM((M, N)).to(device)
    model.load_state_dict(torch.load(weights_file))

    # load mean and std
    matfile = loadmat(meanstd_file)
    test_mean = np.array(matfile['mean_test'])
    test_std = np.array(matfile['std_test'])

    # normalize images using the mean and std
    Images_norm = np.zeros(Images.shape, dtype=np.float32)
    for i in range(K):
        Images_norm[i, ...] = project_01(Images[i, ...])
        Images_norm[i, ...] = normalize_im(Images_norm[i, ...], \
                test_mean, test_std)
    Images_norm = Images_norm.reshape([K, 1, *Images.shape[1:]])

    # make prediction (and time it)
    st = time.time()
    model.eval()
    predicted_density = np.zeros(Images_norm.shape, dtype=np.float32)
    for i in range(K):
        Image_norm = torch.from_numpy(Images_norm[i, ...]).to(device)[np.newaxis]
        predicted_density[i, ...] = model(Image_norm).detach().cpu().numpy()
        Image_norm = Image_norm.cpu() # move back to cpu regardless
    et = time.time()
    print(et-st)

    # project to non-negative orthant
    predicted_density[predicted_density < 0] = 0.0

    # resulting sum images
    WideField = np.squeeze(np.sum(Images_norm, axis=0))
    Recovery = np.squeeze(np.sum(predicted_density, axis=0))

    # plot the sum images
    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
    ax[0].imshow(WideField)
    ax[0].set_title("Wide Field")
    ax[0].axis('off')
    ax[1].imshow(Recovery)
    ax[1].set_title("Sum of Predictions")
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()

    # save predictions in mat file
    mdict = {'Recovery': Recovery}
    savemat(savename, mdict)

    # save each frame for debugging
    if debug:
        mdict = {"Predictions": predicted_density}
        savemat(savename + "_predictions.mat", mdict)

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', \
            help="path to tiff stack for reconstruction")
    parser.add_argument('--weights_name', \
            help="path to the trained model weights as hdf5-file")
    parser.add_argument('--meanstd_name', \
            help="path to the saved normalization factors as m-file")
    parser.add_argument('--savename', type=str, \
            help="path for saving the Superresolution reconstruction matfile")
    parser.add_argument('--upsampling_factor', type=int, default=8, \
            help="desired upsampling factor")
    parser.add_argument('--debug', type=int, default=0, \
            help="boolean (0/1) for saving individual predictions")
    args = parser.parse_args()

    # run testing with given arguements
    test_model(abspath(args.datafile), abspath(args.weights_name), \
            abspath(args.meanstd_name), abspath(args.savename), \
            args.upsampling_factor, args.debug)

