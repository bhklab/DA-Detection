"""
This is the main script in the DA-Detection package which classifies a set of CT images
as containing strong, weak, or no dental artifacts (DA).
"""
import os
import numpy as np

from data_loading.data_loader import DataLoader
from SBD.classify import Classifier
from CNN.classify import classify_img


def setup_data_loading(csv_path, img_path, img_type='dicom') :
    """ This function initializes a data loader object, defined in data_loading.
    This object returns the image given a particular index.
    """
    data_set = DataLoader(img_dir, label_path, img_suffix, file_type="dicom")


def setup_SBD(dicom_path, da_labels_path) :
    """ This function handles all setup for sinogram-based DA detection (SBD).
    It takes the path to the DA labels, the path to the directory containing
    .DICOM files (each )
    """


def run_SBD() :
    pass

def run_CNN() :

    pass

def decision_network(sbd_predictions, cnn_predictions) :
    pass
