# import all the necessary packages
import torch
from deeplogger.model import UNet
import matplotlib.pyplot as plt
import pickle
import numpy as np
from skimage.transform import iradon
from scipy.io import savemat
from skimage.color import rgb2hsv

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_as_matlab_array(im_num, filename):
    input_sample = torch.load('/home/alexis/Documents/DATA/DeepLogger/SKB/training_data/ID0000' + str(im_num) + '.pt')
    data = {}
    data['image'] = input_sample[0].detach().numpy()
    data['mask'] = input_sample[1].detach().numpy()
    savemat(filename + '_image.mat', data)
    


def plot_radon_transforms(im_num):
    input_sample = torch.load('/home/alexis/Documents/DATA/DeepLogger/SKB/training_data/ID0000' + str(im_num) + '.pt')
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

def plot_data_and_mask(im_num):
    input_sample = torch.load('/home/alexis/Documents/DATA/DeepLogger/SKB/training_data/ID0000' + str(im_num) + '.pt')
    image = input_sample[0]
    mask = input_sample[1]
    plt.figure(figsize=(20,10))
    ax1 = plt.subplot(121)
    ax1.imshow(image)
    ax1.set_title('Borehole image data')
    ax2 = plt.subplot(122)
    ax2.imshow(mask, cmap='bone')
    ax2.set_title('Mask used to train model')

def plot_image_comparison(im_num):
    input_sample = torch.load('/home/alexis/Documents/DATA/DeepLogger/SKB/training_data/ID0000' + str(im_num) + '.pt')
    rgb_image = input_sample[0].squeeze().detach().numpy()
    hsv_image = rgb2hsv(rgb_image)
    mask = input_sample[1].squeeze().detach().numpy()
    plt.figure(figsize=(20,20))
    ax1 = plt.subplot(441)
    im1 = ax1.imshow(rgb_image[:,:,0])
    ax2 = plt.subplot(442)
    im2 = ax2.imshow(rgb_image[:,:,1])
    ax3 = plt.subplot(443)
    im3 = ax3.imshow(rgb_image[:,:,2])
    ax4 = plt.subplot(444)
    im4 = ax4.imshow(mask)
    ax5 = plt.subplot(445)
    im5 = ax5.imshow(hsv_image[:,:,0])
    ax6 = plt.subplot(446)
    im6 = ax6.imshow(hsv_image[:,:,1])
    ax7 = plt.subplot(447)
    im7 = ax7.imshow(hsv_image[:,:,2])
    ax8 = plt.subplot(448)
    im8 = ax8.imshow(mask)
    ax1.set_title('Red')
    ax2.set_title('Green')
    ax3.set_title('Blue')
    ax4.set_title('Ground Truth')
    ax5.set_title('Hue')
    ax6.set_title('Saturation')
    ax7.set_title('Value')
    ax8.set_title('Ground Truth')    
    im1.set_cmap('Reds')
    im2.set_cmap('Greens')
    im3.set_cmap('Blues')
    im4.set_cmap('Greys')
    im5.set_cmap('Oranges')
    im6.set_cmap('Purples')
    im7.set_cmap('Greys')
    im8.set_cmap('Greys')


#04_30_UNet_deeplogger_RGB-epoch-525.pt
folder = '04_30'
data_type = 'RGB'
model_name = folder + "_UNet_deeplogger"
saved_models_path = '/home/alexis/Documents/DATA/DeepLogger/models/' + folder + '/'


config = load_obj(saved_models_path + model_name + '_' + data_type + '_config.p')
epoch = 525
    
    
plt.figure()
training_loss = np.array(config['training_losses'])
validation_loss = np.array(config['validation_losses'])
ax1 = plt.subplot(2,1,1)
ax1.plot(training_loss)

ax2 = plt.subplot(2,1,2)
ax2.plot(validation_loss)


#
#with open(model_name + '_config.p', 'rb') as fp:
#    config = pickle.load(fp)


def dice_coefficient(y_pred, y_true,smooth):
    y_pred = y_pred[:, 0].contiguous().view(-1)
    y_true = y_true[:, 0].contiguous().view(-1)
    intersection = (y_pred * y_true).sum()
    dsc = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    return dsc.item()


device = torch.device("cpu")
forward_model = UNet().to(device)
forward_model = forward_model.double()
forward_model.load_state_dict(torch.load(saved_models_path + model_name + '_RGB-epoch-' + str(epoch) + '.pt', map_location=torch.device('cpu')))
forward_model.eval()



for im_num in [10,20,30,40,50]:
    input_sample = torch.load('/home/alexis/Documents/DATA/DeepLogger/SKB/RGB_training_data/ID0000' + str(im_num) + '.pt')
    prediction_evaluation = forward_model(input_sample[0].unsqueeze(0).double())
    dice_coefficient(prediction_evaluation.squeeze(), input_sample[1],1)
    plt.figure(figsize=(20,10))
    ax1 = plt.subplot(131)
    ax1.imshow(input_sample[0].squeeze().detach().numpy())
    ax1.set_title('Borehole image data')
    ax1.set_xlabel('Borehole azimuth (degrees)')
    ax1.set_ylabel('Image depth (mm)')
    ax2 = plt.subplot(132)
    ax2.imshow(input_sample[1].squeeze().detach().numpy())
    ax2.set_title('Mask used to train model')
    ax2.set_xlabel('Borehole azimuth (degrees)')
    ax2.set_ylabel('Image depth (mm)')
    ax3 = plt.subplot(133)
    ax3.imshow(prediction_evaluation.squeeze().detach().numpy())
    ax3.set_title('Model prediction')
    ax3.set_xlabel('Borehole azimuth (degrees)')
    ax3.set_ylabel('Image depth (mm)')
    plt.savefig('image_comparison' + str(im_num) + '.png')




#
#save_as_matlab_array(38, 'borehole_data')
#
#
#plot_radon_transforms(38)
#plot_data_and_mask(38)
#
#
### Display colormaps
#plot_image_comparison(40)
#plot_image_comparison(10)



