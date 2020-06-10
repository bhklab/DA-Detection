# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:02:59 2019

@author: welchm
"""
import io
import os
import torch
import math
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import SimpleITK as sitk
from skimage.transform import resize

import pandas as pd




def Softmax(x):
    #Calculate probability for each class using softmax function
    #https://nolanbconaway.github.io/blog/2017/softmax-numpy

    e_x = np.exp(x-np.max(x))
    return e_x/e_x.sum(axis = 0)

def softmax() :
    sm = torch.nn.Softmax()
    sm = sm.cuda()
    return sm

def GetPredictions(image, net):
    #Input image to net to get DA classification prediction
    #Returns:
    # 1) maximum of the net outputs as the predicted label
    # 2) softmax probabilities for each class
    image = Variable(image)
    output = net(image)

    # outputs_softmax = Softmax(output.data.numpy()[0])
    sm = softmax()
    outputs_softmax = sm(output.data[0])
    _, predicted = torch.max(output.data, 1)

    # softmax_prob = np.array(outputs_softmax)
    softmax_prob = outputs_softmax
    predicted_label = predicted[0]
    # predicted_label = predicted.numpy()[0]

    return predicted_label, softmax_prob

class Net(nn.Module):
    #CNN toplogy definition
    def __init__(self, output_dim):
        super(Net, self).__init__()

        self.pool = nn.MaxPool3d(2, 2)
        self.LRelu = nn.LeakyReLU(0.01)

        self.conv1 = nn.Conv3d(1, 4, 5, padding=2)
        self.conv1_bn = nn.BatchNorm3d(4)

        self.conv2 = nn.Conv3d(4, 8, 3, padding=1)
        self.conv2_bn = nn.BatchNorm3d(8)

        self.conv3 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv3_bn = nn.BatchNorm3d(16)

        self.conv4 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv4_bn = nn.BatchNorm3d(32)

        self.conv5 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv5_bn = nn.BatchNorm3d(64)

        self.avgPool = nn.AvgPool3d(2, 2)

        self.fc3 = nn.Linear(64 * 8 * 8 * 8, output_dim)




    def forward(self, input1):
        input1 = self.pool(self.conv1_bn(self.LRelu(self.conv1(input1))))
        input1 = self.pool(self.conv2_bn(self.LRelu(self.conv2(input1))))
        input1 = self.pool(self.conv3_bn(self.LRelu(self.conv3(input1))))
        input1 = self.pool(self.conv4_bn(self.LRelu(self.conv4(input1))))
        input1 = self.conv5_bn(self.LRelu(self.conv5(input1)))
        input1 = self.avgPool(input1)
        input1 = input1.view(-1, 64 * 8 * 8 * 8)
        input1 = self.fc3(input1)

        return input1

def LoadNet(path, name, on_gpu=True):
    #Load and define network parameters.
    #CNN file assumed to be in same directory as DAClassification.py module
    nclasses = 2
    net = Net(output_dim = nclasses)

    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.0001

    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=momentum, weight_decay = weight_decay)

    checkpoint = torch.load(os.path.join(path, name), map_location=lambda storage, loc:storage)
    optimizer.load_state_dict(checkpoint['optimizer'])

    net.load_state_dict(checkpoint['state_dict'])

    if on_gpu :
        net = net.cuda()

    net.eval()

    return net

def Resample(image):

    #Resample image to common resolution of 1x1x1
    new_spacing = [1,1,1]

    #Set up SitK resampling image filter
    rif = sitk.ResampleImageFilter()
    rif.SetOutputSpacing(new_spacing)
    rif.SetOutputDirection(image.GetDirection())

    #Get original image size and spacing
    orig_size = np.array(image.GetSize(), dtype = np.int)
    orig_spacing = np.array(image.GetSpacing())

    #Calculate new image size based on current size and desired spacing.
    new_size = np.ceil(orig_size*(orig_spacing/new_spacing)).astype(np.int)
    new_size = [int(s) for s in new_size]

    #Set up SitK resampling parameters
    rif.SetSize(new_size)
    rif.SetOutputOrigin(image.GetOrigin())
    rif.SetOutputPixelType(image.GetPixelID())
    rif.SetInterpolator(sitk.sitkLinear)

    #Resample image and generate numpy array from image
    resampledImage = rif.Execute(image)
    imageArray = sitk.GetArrayFromImage(resampledImage)


    # imageArray = image

    return imageArray

def Resize(image):
     #Resize image to isotropic dimensions (256x256x256)
     #Pad image to retain aspect ratio

     #Generate isotropic array of zeros based on maximum dimension of image
     pad = np.zeros((3,1))
     pad[0,0] = max(image.shape) - image.shape[0]
     pad[1,0] = max(image.shape) - image.shape[1]
     pad[2,0] = max(image.shape) - image.shape[2]

     paddedImage = np.zeros((max(image.shape),max(image.shape),max(image.shape)))

     #Pad image
     paddedImage = np.pad(image, ((int(math.ceil(pad[0,0]/2)), int(math.floor(pad[0,0]/2))),(int(math.ceil(pad[1,0]/2)), int(math.floor(pad[1,0]/2))),(int(math.ceil(pad[2,0]/2)), int(math.floor(pad[2,0]/2)))), 'constant', constant_values=0)

     #Resize padded image to desired size 256
     size_new = 256
     image_resized = resize(paddedImage, (size_new, size_new, size_new), preserve_range = True)

     return image_resized


def ToTensor(image):
    #Convert numpy array representation of image to tensor representation for usage in Pytorch CNN
    """Convert ndarrays in sample to Tensors."""
    #Expand dimensions to account for single image (versus expected batch) and missing colour channel.
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)

    dtype = torch.cuda.FloatTensor
    # dtype = torch.FloatTensor
    return torch.from_numpy(image).type(dtype)
    # return torch.from_numpy(image.copy()).float()


def read_dicom_image(dicom_path) :
    """Return SITK image given the absolute path
    to a DICOM series."""
    reader = sitk.ImageSeriesReader()
    # path is the directory of the dicom folder
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    return image

def LoadImage(image_path, on_gpu=True):


    #Load image for classification.
    #Assumes nrrd file capable of reading with SitK library
    #Returns tensor representation of image
    # image = sitk.ReadImage(image_path)
    image = read_dicom_image(image_path)

    # BHKlab implementation: Load npy image with np.load
    # image = np.load(image_path)

    #Transform image using resample, resize functions
    resamp_image = Resample(image)
    resize_image = Resize(resamp_image)

    #Generate tensor of image
    image_tensor = ToTensor(resize_image)

    return image_tensor
