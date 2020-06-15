import os
import numpy as np
import SimpleITK as sitk

def read_npy_image(file_path) :
    img = np.load(file_path)
    return img



def read_nrrd_image(nrrd_file_path) :
    image = sitk.ReadImage(nrrd_file_path)

    # Resize the image
    # resamp_img = resample_image(image, new_spacing=[1,1,1])
    # if as_array
    #     imageArray = sitk.GetArrayFromImage(image)
    return imageArray





def read_dicom_image(dicom_path) :
    """Return SITK image given the absolute path
    to a DICOM series."""
    reader = sitk.ImageSeriesReader()
    # path is the directory of the .DICOM folder
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Resize the image
    # resamp_img = resample_image(image, new_spacing=[1,1,1])

    # Comvert image to np array
    # if as_array :
    #     image = sitk.GetArrayFromImage(image)

    return image
