# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:05:04 2019

@author: welchm
"""

#Example script for using DAClassification.py module for 
#      classification of Dental artifact presence in single H&N CT image. 
import DAClassification
import os
import sys

#path to image
image_path = 'Q:/RTxomics/sorted/NRRD/0435496/20110117_/Reconstructions/2 Neck 2.0 CE.nrrd'

#Path to CNN file assumed to be in same directory as DAClassification.py
net_path = os.path.abspath(os.path.dirname(sys.argv[0])) 
net_name = 'testCheckpoint.pth.tar'

#Load and transform image. 
# 1) Image loading achieved using SimpleITK. 
# 2) Image resampled to isotropic voxel size 1x1x1
# 3) Resampled image resized to 256x256x256 array while retaining aspect ratio
# 4) Resampled and resized image numpy array formated to Pytorch tensor
image = DAClassification.LoadImage(image_path)

#Load CNN for DA Classification. 
#Uses five 3D convolutional layers to achieve a final machine generated 
#      features of size 8x8x8
net = DAClassification.LoadNet(net_path, net_name)

#Apply net to image for DA classification prediction
#Returns:
# 1) predicted_label = class associated maximum value returned by the net
# 2) softmax_prob = array of probabilities 
#       a) softmax_prob[0] = probability of DA- image (ie. no DA)
#       b) softmax_prob[1] = probability of DA+ image (ie. DA present))
predicted_label, softmax_prob = DAClassification.GetPredictions(image, net)