'''
A script which finds the locations of dental artifacts
in CT scans. The script loads images which have already
been identified as containing DAs (either through manual
labelling or using an automated classifier).

The algorithm works by doing the following for each 3D image:
  1. Clipping the full 3D image's intensity range to between
     max(image) and max + 200 HU.
  2. Calculate the per-slice standard deviation.
  3. Perform peak detection on the per-slice std:
     - If no peaks are found in the per-slice std,
       3.1. Then renormalize the original image with the
            lower HU limit decreased by 20 HU.
       3.2. Repeat steps 2 and 3.
     - Otherwise, the indeces of the peaks are saved as the
       locations of the artifacts.
'''
import os
import numpy as np
import pandas as pd
import json
import time

import scipy.stats as stat

from scipy.signal import find_peaks


class GetLocation(object):
    """Finds the location of artifacts in a patient"""
    def __init__(self, args):
        super(GetLocation, self).__init__()
        # Data directories
        self.img_dir = "/cluster/projects/radiomics/Temp/RADCURE-npy/img"
        # self.label_path = "/cluster/home/carrowsm/data/radcure_DA_labels.csv"
        self.label_path = "/cluster/home/carrowsm/MIRA/radcure_DA_labels.csv"

        # Load CSV containing all image labels
        label_df = pd.read_csv(self.label_path, na_values=["","nan"], dtype=str, index_col="p_index")

        # Print some stats about labels
        n_s = len(label_df[label_df["has_artifact"] == "2" ])
        n_w = len(label_df[label_df["has_artifact"] == "1" ])
        n_n = len(label_df[label_df["has_artifact"] == "0" ])
        print(f'Images labelled strong: {n_s}')
        print(f'Images labelled weak:   {n_w}')
        print(f'Images labelled none:   {n_n}')
        print(f'Finding the z-index of artifacts in {len(label_df[label_df["has_artifact"] != "0" ])} images')

        # Keep only images with positive artifact labels
        self.labels = label_df[label_df["has_artifact"] != "0" ]


    def detect_peaks(self, x) :
        ''' Detects peaks in a 1D array x.'''
        m = np.median(x)
        std = np.std(x)
        h = m + 1.5*std

        peak_indices, _ = find_peaks(x,
                                     distance=None,
                                     threshold=None,
                                     height=h,
                                     prominence=None)
        return peak_indices

    # Algorithm to iteratively decrease the lower normalization bound until artifact is found
    def norm_std(self, X, MIN, MAX):
        """Takes an array of images slices and
        normalizes it, then returns the std deviation
        per slice."""
        clipped = np.clip(X, MIN, MAX)
        X = (clipped - MIN) / (MAX - MIN)

        ### Calculate per-slice std  ###
        return np.std(X, axis=(1,2))

    def find_art_loc(self, X, min_range=200.0) :
        """ X is a stack of CT images"""
        ### Normalize (first attempt)  ###
        max_ = np.max(X) + 200.0   # We want a range of [max+200, max]
        min_ = max_ - min_range    # Where the lower bound decreases every iteration

        ### Normalize and calculate per-slice std  ###
        z_std = self.norm_std(X, min_, max_)

        ### Try to find artifact location ###
        peaks = self.detect_peaks(z_std)

        ### Calculate area under z_std curve ###
        # integral = np.trapz(z_std, dx=1)
        integral = 0

        ### If no peaks were found, decrease
        #  lower limit and do it again      ###
        if len(peaks) == 0 :
            new_min_range = min_range + 50.0
            print("Trying again, HU range: ", max_ - new_min_range, max_)
            return self.find_art_loc(X, min_range=new_min_range)
        else :
            return peaks, integral

    def run_on_data(self) :
        # Test on labelled data
        preds = {}
        integrals = []
        times = []


        for patient_id in self.labels["patient_id"].values :
            # t0 = time.time()
            file_name = patient_id + "_img.npy"

            print(f"Finding DA location in patient {patient_id}")

            try :
                # Load image
                # t1 = time.time()
                X = np.load(os.path.join(self.img_dir, file_name))

                X = X.astype(np.int16)    # Make all pixels 16-bit integers
                # print(time.time() - t1)
            except :
                preds[patient_id] = ["LoadingError"]
                continue

            # Detect peaks
            # t1 = time.time()
            peaks, integral = self.find_art_loc(X)
            # print(time.time() - t1)

            # Add results to dictionary
            preds[patient_id] = peaks.tolist()
            integrals.append(integral)

            # t_elapsed = time.time() - t0
            # print(str(t_elapsed) + "\n")
            # print("#" * 10)
            # times.append(t_elapsed)

        return preds, integrals, times



# Accuracy assessment
def assess_acc(ids, labels, preds, integrals) :
    single_preds = [np.median(preds[key]) for key in preds]
    df = pd.DataFrame({"pid": ids, "label": labels, "pred": single_preds}, dtype=float)

    df["loc_diff"] = df["label"] - df["pred"]
    size = len(ids)

    exact = sum(df["loc_diff"] == 0) / size
    within_5  = sum(df["loc_diff"] < 5) / size
    within_10 = sum(df["loc_diff"] < 10) / size
    within_15 = sum(df["loc_diff"] < 15) / size
    within_20 = sum(df["loc_diff"] < 20) / size
    return [exact, within_5, within_10, within_15, within_20], df


def save_json(preds_dict) :
    json_file = "loc_preds.json"
    ### SAVE SPECIFIC SLICE INDICES CONTAINING ARTIFACTS ###
    # JSON implementation
    print("Saving JSON")
    with open(json_file, 'w') as f :           # Open JSON and write
        json.dump(preds_dict, f)                # new dictionary to file
    print("JSON Saved")



if __name__ == "__main__" :

    print("Finished importing")

    args = []
    model = GetLocation(args)
    print("Model loaded")


    preds, integrals, times = model.run_on_data()
    print("All images labelled")

    print("Assessing accuracy")
    patient_ids = model.labels["patient_id"].values
    truth = model.labels["a_slice"].values

    summary, results_df = assess_acc(patient_ids, truth, preds, integrals)

    print(summary)
    results_df.to_csv("non_sinogram_results.csv", na_rep="nan")

    # Save JSON
    save_json(preds)
