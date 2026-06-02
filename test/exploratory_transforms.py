# import all the necessary packages
import torch
from model import UNet
import matplotlib.pyplot as plt
import pickle
import numpy as np
from skimage.transform import iradon
from scipy.io import savemat
from scipy import fft as spfft
from skimage.color import rgb2hsv


def plot_radon_transforms(im_num):
    input_sample = torch.load(data_path+'ID0000' + str(im_num) + '.pt')
    image = input_sample[0]
    mask = input_sample[1]
    radon_image = iradon(image[:,:,0].detach().numpy().T,theta=np.linspace(0,360,360),circle=False)
    radon_mask = iradon(mask.detach().numpy().T,theta=np.linspace(0,360,360),circle=False)
    fig = plt.figure(figsize=(20,10))
    ax1 = plt.subplot(121)
    im1 = ax1.imshow(radon_image, cmap='bone')
    ax1.set_title('iRadon transform of borehole image data')
    fig.colorbar(im1, orientation='horizontal')
    ax2 = plt.subplot(122)
    im2 = ax2.imshow(radon_mask, cmap='bone')
    ax2.set_title('iRadon transform of mask used to train model')
    fig.colorbar(im2, orientation='horizontal')

def plot_fft_transforms(im_num):
    input_sample = torch.load(data_path+'ID0000' + str(im_num) + '.pt')
    image = input_sample[0]
    mask = input_sample[1]
    fft_image = spfft.fft2(image[:,:,0].detach().numpy().T)
    fft_mask = spfft.fft2(mask.detach().numpy().T)
    fig = plt.figure(figsize=(20,10))
    ax1 = plt.subplot(221)
    im1 = ax1.imshow(np.log(np.real(fft_image)), cmap='Reds')
    ax1.set_title('log(real(fft)) transform of borehole image data')
    fig.colorbar(im1, orientation='horizontal')
    ax2 = plt.subplot(222)
    im2 = ax2.imshow(np.log(np.real(fft_mask)), cmap='Reds')
    ax2.set_title('log(real(fft)) transform of mask used to train model')
    fig.colorbar(im2, orientation='horizontal')
    ax3 = plt.subplot(223)
    im3 = ax3.imshow(np.log(np.imag(fft_image)), cmap='Reds')
    ax3.set_title('log(imag(fft)) transform of borehole image data')
    fig.colorbar(im3, orientation='horizontal')
    ax4 = plt.subplot(224)
    im4 = ax4.imshow(np.log(np.imag(fft_mask)), cmap='Reds')
    ax4.set_title('log(imag(fft)) transform of mask used to train model')
    fig.colorbar(im4, orientation='horizontal')
