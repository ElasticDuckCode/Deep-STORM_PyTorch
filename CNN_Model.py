#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# project image min and max values to 0 and 1
def project_01(im):
    im = np.squeeze(im)
    min_val = im.min()
    max_val = im.max()
    return (im - min_val) / (max_val - min_val)

# normalize image by given mean and std
def normalize_im(im, dmean, dstd):
    im = np.squeeze(im)
    im_norm = np.zeros(im.shape, dtype=np.float32)
    im_norm = (im - dmean)/dstd
    return im_norm

# define 2D gaussian filter function to mimic 
# MATLAB's fspecial('gaussian', [shape], [sigma])
def matlab_style_gauss2D(shape=(7,7), sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x**2 + y**2) / (2.0*sigma**2))
    h.astype('float32')
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = h * 2.0
    h = h.astype('float32')
    return h

# prepare the filter used in Deep-STORM
psf_heatmap = torch.tensor(matlab_style_gauss2D(shape=(7, 7), sigma=1)).to(device)
gfilter = torch.reshape(psf_heatmap, [1, 1, 7, 7]).to(device)

# Deep-STORM Loss function
def l1l2loss(heatmap_true, spikes_pred):
    # use spikes to create heatmaps
    heatmap_pred = F.conv2d(spikes_pred, gfilter, padding=3) # why same isn't working
    # heatmap l2 error
    loss_heatmap = F.mse_loss(heatmap_true, heatmap_pred)
    # spikes l1 error
    loss_spikes = F.l1_loss(spikes_pred, torch.zeros_like(spikes_pred))
    return loss_heatmap + loss_spikes

class DeepSTORM(nn.Module):
    def __init__(self, input_shape):
        super(DeepSTORM, self).__init__()

        # "Encoder" Part of Network
        self.features1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.features2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.features3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2)
        self.features4 = nn.Sequential(
            nn.Conv2d(128, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # "Decoder" Part of Network
        self.up5 = nn.Upsample(scale_factor=2)
        self.features5 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up6 = nn.Upsample(scale_factor=2)
        self.features6 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up7 = nn.Upsample(scale_factor=2)
        self.features7 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.prediction = nn.Conv2d(32, 1, 1, padding=0, bias=False)

    def forward(self, x):
        x = self.features1(x)
        x = self.pool1(x)
        x = self.features2(x)
        x = self.pool2(x)
        x = self.features3(x)
        x = self.pool3(x)
        x = self.features4(x)
        x = self.up5(x)
        x = self.features5(x)
        x = self.up6(x)
        x = self.features6(x)
        x = self.up7(x)
        x = self.features7(x)
        y = self.prediction(x)
        return y

if __name__ == "__main__":
    network = DeepSTORM((16,16))
    print(network)
