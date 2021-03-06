import numpy as np
import pandas as pd
import os
from config import get_args

''' This contains the main data loading and preprocessing functions.'''


class DataLoader(object):
    """docstring for DataLoader."""
    def __init__(self, file_type="npy", args):
        # super(DataLoader, self).__init__()

        self.img_dir, self.img_suffix = args.img_dir, args.img_suffix
        self.saving, self.log_dir = args.logging, args.logdir
        self.calc_acc = args.calc_acc
        self.label_dir = args.label_dir
        self.file_type = file_type

        # Get a list containing the path to each file
        self.patient_list = self.get_patient_list()

        self.dataset_length = len(self.patient_list) # Total number of patients



    def get_patient_list(self) :
        ''' Function returns a pandas data frame containing all the patients
            and their file names'''

        # # If we are using labels, get the DF from the label DF
        # if self.calc_acc :
        #     # Load data as a pandas DataFrame
        #     df = pd.read_csv(self.label_dir, index_col="p_index",
        #                      dtype=str, na_values=['nan', 'NaN', ''])
        #
        #     # Uncomment the next two lines if labels are incomplete
        #     first_entry = df["has_artifact"].first_valid_index()
        #     last_entry = df["has_artifact"].last_valid_index()
        #     df = df.loc[first_entry : last_entry]
        #
        #     patient_list = df["patient_id"].values
        #     label_list = df["has_artifact"].values.astype(int)
        #     return patient_list, label_list
        # else :
        #     # Make a list of patient IDs from scratch
        #     patient_list = []
        #     for file in os.listdir(self.img_dir) :
        #          # Get the ID from the filename
        #         patient_list.append(file.split("_")[0])
        #     return patient_list, None

        # Get a df from the CSV of image DA labels
        df = pd.read_csv(self.label_dir, index_col="p_index",
                         dtype=str, na_values=['nan', 'NaN', ''])
        df = df.set_index("patient_id")
        df["file_path"] = np.zeros(len(df)) * np.nan # Initialize with list of nans

        # Get the path to the file for each patient
        for id in df.index :
            file_name = str(id) + self.image_suffix + "." + self.file_type
            if self.file_type == "npy" or self.file_type == "nrrd" :
                full_path = os.path.join(self.img_dir, file_name)

            elif self.file_type == "dicom" :
                full_path = self.get_dicom_path(os.path.join(self.img_dir, id))

            else :
                raise(NotImplementedError(f"File type {self.file_type} not supported. Must be 'nrrd', 'npy', or 'dicom'."))

            df.loc[id, "file_path"] = full_path

        return df.loc[:, "file_path"].values

    def get_dicom_path(self, path) :
        """Given a path like
        /cluster/projects/radiomics/RADCURE-images/1227096/,
        find the actual DICOM series directory"""
        dicom_path = None
        for root, dirs, files in os.walk(path, topdown=True):
            for name in dirs:
                if name.endswith(".DICOM") :
                    dicom_path = os.path.join(root, name)
        return dicom_path


    def getitem(self, index):
        '''Load the images for the patient corresponding to index
            in the patient_list'''

        pid = self.patient_list[index]
        label = self.label_list[index] if self.calc_acc else None

        # Get the full path to the npy file containing this patient's scans
        file_name = str(pid) + self.img_suffix
        full_path = os.path.join(self.img_dir, file_name)

        # Load the np array representing the patient's image Stack
        if self.file_type == 'npy' :
            img = np.load(full_path, mmap_mode='r')
        elif self.file_type == 'nrrd' :
            img = 

        return img, label


    def __len__(self):
        return self.dataset_length




if __name__ == '__main__' :

    args, unparsed = get_args()

    dl = DataLoader(args)
