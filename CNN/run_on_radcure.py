import sys
import os
import pandas as pd
import numpy as np
import DAClassification
import torch

import logging
import time



def get_dicom_path(path) :
    """Given a path like
    /cluster/projects/radiomics/RADCURE-images/1227096/,
    find the actual DICOM series directory"""
    dicom_path = None
    for root, dirs, files in os.walk(path, topdown=True):
        for name in dirs:
            if name[-6 : ] == ".DICOM" :
                dicom_path = os.path.join(root, name)
                # logging.info("DICOM Found")
    return dicom_path




def classify_img(img_path, net=None, on_gpu=True) :
    """
    Run Mattea's code from RUNME.py. Classifies a single image using her CNN.
    """
    #Path to CNN file assumed to be in same directory as DAClassification.py
    # net_path = os.path.abspath(os.path.dirname(sys.argv[0]))
    # net_name = 'testCheckpoint.pth.tar'

    #Load and transform image.
    # 1) Image loading achieved using SimpleITK.
    # 2) Image resampled to isotropic voxel size 1x1x1
    # 3) Resampled image resized to 256x256x256 array while retaining aspect ratio
    # 4) Resampled and resized image numpy array formated to Pytorch tensor
    image = DAClassification.LoadImage(img_path, on_gpu=on_gpu)


    #Load CNN for DA Classification.
    #Uses five 3D convolutional layers to achieve a final machine generated
    #      features of size 8x8x8
    # net = DAClassification.LoadNet(net_path, net_name)

    #Apply net to image for DA classification prediction
    #Returns:
    # 1) predicted_label = class associated maximum value returned by the net
    # 2) softmax_prob = array of probabilities
    #       a) softmax_prob[0] = probability of DA- image (ie. no DA)
    #       b) softmax_prob[1] = probability of DA+ image (ie. DA present))

    if on_gpu :
        image = image.cuda() # Put the image tensor on a GPU

    predicted_label, softmax_prob = DAClassification.GetPredictions(image, net)


    return predicted_label, softmax_prob


if __name__ == "__main__" :

    # Logging
    logging.basicConfig(level=logging.DEBUG)

    # NETWORK LOADING #
    net_path = os.path.abspath(os.path.dirname(sys.argv[0]))
    net_name = 'testCheckpoint.pth.tar'
    network = DAClassification.LoadNet(net_path, net_name, on_gpu=True)

    # GPU Computing
    logging.info(f"{torch.cuda.is_available()}")
    torch.cuda.set_device(0)    # Set the device to a GPU
    network.cuda()              # Put the model on the GPU
    logging.info("Network Loaded")
    #-###############-#


    # IMAGE LOADING #
    # Load all labelled radcure images
    # img_dir = "/cluster/projects/radiomics/Temp/RADCURE-npy/img"   # NPY images
    img_dir = "/cluster/projects/radiomics/RADCURE-images"         # DICOM images
    # img_dir = "/cluster/projects/radiomics/Temp/NEW/itk/img"         # NRRD images
    label_dir = "/cluster/home/carrowsm/MIRA/radcure_DA_labels.csv"  # Labels
    labels_df = pd.read_csv(label_dir, index_col="p_index",
                            dtype=str, na_values=['nan', 'NaN', ''])
    logging.info("Labels Loaded")

    #-###############-#



    preds = []              # Predictions made by the CNN (same order as labels_df)
    prob0 = []              # Softmax probabilities from the CNN (same order as labels_df)
    prob1 = []

    # Iteratively classify each patient with the CNN
    for index in labels_df.index :
        patient_id = labels_df.loc[index, "patient_id"]
        # img_path = dicoms[index]

        # Get the SITK image from DICOM

        img_path = get_dicom_path(os.path.join(img_dir, patient_id))

        # img_path = os.path.join(img_dir, patient_id+".nrrd")
        if img_path is None :
            preds.append(np.nan)
            prob0.append(np.nan)
            prob1.append(np.nan)
            continue   # Skip this patient if the file does not exist


        # Get the image's label
        truth = labels_df.loc[index, "has_artifact"]


        # Run the image through the CNN classifier
        t1 = time.time()
        predicted_label, softmax_prob = classify_img(img_path, net=network)
        t_f = time.time() - t1

        logging.info(f"{patient_id} prediction: {predicted_label*2}, label: {truth} in {t_f} s")

        preds.append(predicted_label.item())
        prob0.append(softmax_prob[0].item())
        prob1.append(softmax_prob[1].item())

    # Add results to the data frame and save
    labels_df["CNN_preds"] = preds
    labels_df["CNN_probs0"] = prob0
    labels_df["CNN_probs1"] = prob1

    # rename our columns
    labels_df = labels_df.rename(columns={"has_artifact": "manual_artifact_status", "a_slice":"manual_artifact_location"})

    labels_df.to_csv("RadCure_Predictions.csv")
