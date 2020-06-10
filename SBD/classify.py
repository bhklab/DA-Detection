import os
import json
import time
import numpy as np
import pandas as pd
from data_loader import DataLoader
from config import get_args


from scipy.ndimage.morphology import binary_fill_holes as bf
from skimage.filters import threshold_otsu, gaussian

from skimage.transform import radon


from scipy.signal import find_peaks

import multiprocessing

# Surpress warnings... probably not a great idea
import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.DEBUG)
logging.info("Finished importing modules")

'''
This script classifies images in the RadCure dataset which contain 'strong'
metal artifact streaks.

The program processes each patient slice-by-slice, and picks out slices with
where there is very strong evidence of artifacts.
'''

class Classifier(object):
    """ Class containing functions to run automatic (non-DL) per-slice
        classification of images with artifacts."""
    def __init__(self, args, data_loader):

        self.data_loader = data_loader

        # Test mode. If true, do only a few iterations
        self.test_mode = args.test

        # File which will contain the predictions made by this class
        self.log_file = os.path.join(args.logdir, "preds")

        # File containing the 'true' class labels
        self.true_labels_dir = args.label_dir

        self.sigma = 10 # Width of Gaussian for blur
        self.t1 = 0.01  # Threshold for removal of body
        self.t2 = 0.02  # Threshold after body is removed

        # Dictionary containing each patient and the slice indeces of the artifacts
        self.art_slices = {}
        self.class_list = [] # list containing binary classifications
                             # (1=has artifact), same order as patient_list

    def save_patient(self, patient_id, indices, save_file) :
        ''' Save the indices which contain artifacts for one patient.
            save_file should be a path and file name without a file extension.
            This function will save one txt and one csv.'''
        log_txt = save_file + ".txt"
        log_csv = save_file + ".csv"

        # TXT Implementation
        with open(log_txt, 'a') as f :
            new_line = "{}:{}\n".format(patient_id, str(indices))
            f.write(new_line)

        ### APPEND BINARY PREDICTION TO CSV ###
        with open(log_csv, 'a') as f :
            label = 1 if len(indices) > 0 else 0
            new_line = "{},{}\n".format(patient_id, str(label))
            f.write(new_line)

    def save_all(self, results) :
        ''' Function which saves makes results into a dictionary and saves
            this dict to a JSON. Dictionary has this form:
        {"patient1_ID": [slice_index1, slice_index2], # Patient1 has 2 slices with artifacts
         "patient2_ID": None,                         # Patient2 has no artifacts
         "patient3_ID": [slice_index1]}               # Patient3 has 1 slice with artifacts
        '''
        json_file = self.log_file + ".json"
        csv_file  = self.log_file + ".csv"

        data_dict = {}
        i, pid, y_n = [], [], [] # index, patient_id, boolean prediction

        # Make data into a dictionary
        for result in results :
            index, patient_id, binary, indices = result

            # Save all in indices containing artifacts (as a list)
            data_dict[patient_id] = list(indices)

            # Get patient's index, patient_id, binary prediction
            i.append(int(index))
            pid.append(str(patient_id))
            y_n.append(int(binary))

        # Make pandas dataframe with binary predictions
        logging.info("Creating CSV file")
        data_df = pd.DataFrame(data={"patient_id": pid, "prediction": y_n},
                               index=i, dtype=str)



        ### SAVE A CSV SUMMARIZING WHICH PATIENTS HAVE ARTIFACTS ###
        logging.info("Saving CSV")
        data_df.to_csv(csv_file)
        logging.info("CSV saved")

        ### SAVE SPECIFIC SLICE INDICES CONTAINING ARTIFACTS ###
        # JSON implementation
        logging.info("Saving JSON")
        with open(json_file, 'w') as f :           # Open JSON and write
            json.dump(data_dict, f)                # new dictionary to file
        logging.info("JSON Saved")


        return data_dict, data_df

    def lin_norm(self, array) :
        oldMin, oldMax = np.min(array), np.max(array)
        newMin, newMax = 0.0, 0.001
        return (array - oldMin) * ((newMax - newMin)/(oldMax - oldMin)) + newMin

    def nonlin_norm(self, X) :
        oldMin, oldMax = 0., 1.
        newMin, newMax = 0., 1.
        B, a = 0.02, 0.005# B=intensity center, a=intensity gaussian width
        return ( (newMax - newMin) / (1+np.exp(-(X-B)/a)) ) + newMin


    def remove_body(self, image, sig=10, t=0.01) :
        try :
            otsu = threshold_otsu(image)                  # Compute Otsu threshold
        except ValueError :
            print("ALL the pixels are the same number?!")
            # print(image)
            return None
        fill = bf(np.array(image > otsu, dtype=int))  # Fill holes
        gauss_fill = gaussian(fill, sigma=sig)        # Add Gaussian  blur
        fill = np.array(gauss_fill < t, dtype=int)    # Threshold again
        cropped = np.multiply(image, fill)            # Crop out body from raw image
        return cropped

    def detect_peaks(self, x) :
        ''' Detects peaks in a 1D array x.'''
        m = np.median(x)
        std = np.std(x)
        h = m + 4.*std

        peak_indices, _ = find_peaks(x,
                                     distance=None,
                                     threshold=None,
                                     height=h,
                                     prominence=4)
        return peak_indices

    def normalize(self, img, MIN=2000., MAX=3000., mean=1911.15, std=1404.58) :
        # Normalize the image (var = 1, mean = 0)
        img = img.astype(float)
        # print(np.min(img), np.max(img))
        img = np.clip(img, MIN, MAX)
        img = (img - MIN) / (MAX - MIN)
        # img = (img - mean) / (std)
        return img

    def classify(self, inputs) :
        '''Takes one patient's stack of images and classifies it as containing
            'strong' artifacts or no artifact.
            Parameters:
                vars = (patient_id, index)
                patient ID: str
                index: the ordered index corresponding to the patient_id
        '''
        pid, index = inputs[0], inputs[1]

        # logging.info("Classifying patient {}".format(pid))

        # Get the image Data
        stack, label = self.data_loader.getitem(index)
        z_size, x_size, y_size = np.shape(stack)
        # stack = stack[80:-20, 0:350, 50:-50] # Limit the image range
        stack = stack[20:-20, 0:350, 50:-50]
                                    # This removes unwanted common features

        # Convert the image to 16-bit integer
        stack = stack.astype(np.int16)

        # Normalize the image
        stack = self.normalize(stack, MIN=-1000.0, MAX=0.0)

        intensities = []


        # Loop through all images in patient's stack of scans
        for image in stack :
            if np.sum(image) < 1.0e-8 :
                intensities.append(0.0)
                continue   # If the image is entirely black, just go to next image

            # Remove the patient's body from the images
            image = self.remove_body(image)

            # If image is None, just go to next image
            # remove_body() may return None if the slice is identically some value
            if type(image) == type(None) :
                continue

            # Threshold the new image
            image = np.array(image > 0.04, dtype=int)
            # image = self.nonlin_norm(image)
            # image = np.array(image > 0.5, dtype=int)
            # image = np.array(image > 0.02, dtype=int)

            # Get sinogram
            theta = np.linspace(0., 180., 180, endpoint=False)
            sinogram = radon(image, theta=theta, circle=True)
            #
            # # Calculate mean intensity in key region of sinogram
            mean = np.mean(sinogram[120:-120, 40:-40])
            # logging.info(mean)
            # mean = np.mean(image)

            # Append to list of intensities
            intensities.append(mean)

        # Find the slices with artifacts
        indices = self.detect_peaks(intensities)
        # Convert indices to list of ints
        indices = [int(k) for k in indices]

        # Get binary classification
        # (Simply if there were any artifacts or not)
        binary = 2 if len(indices) > 0 else 0


        return index, pid, binary, indices




def parallel_setup(num_cpus, p_list) :
    # Get number of available CPUs
    if num_cpus == None :
        num_cpus = multiprocessing.cpu_count()

    # Make a pool of parallel workers
    pool = multiprocessing.Pool(num_cpus)

    # Create list of patient IDs and their corresponding index
    num_p = len(p_list) # number of patients
    tasks = []
    for index in range(num_p) :
        # Make a list containing the inputs of the classifier for each patient
        tasks.append( (p_list[index], index) )


    logging.info("Parallel Setup Complete")
    logging.info("Classifiying {} patients using {} CPUs.".format(num_p, num_cpus))

    return pool, tasks




if __name__ == '__main__' :

    args, unparsed = get_args()

    # Initialize data loader
    dl = DataLoader(args)
    p_list, l_list = dl.patient_list, dl.label_list   # Ordered list of patient IDs and their labels

    if args.test :
        # If in test mode, restrict data set size
        p_list = p_list[0 : 45]
        l_list = l_list[0 : 45]

    num_cpus = args.ncpu

    # Initialize classifier
    classifier = Classifier(args, dl)

    # Setup Parallel computing
    pool, tasks = parallel_setup(num_cpus, p_list)


    # Start computation in parallel
    t0 = time.time()
    with pool as p :
        results = p.map(classifier.classify, tasks)
    comp_time = time.time() - t0

    # Close Pool and let all the processes complete
    logging.info("Computation finished. Closing pool.")
    # pool.close()
    # pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

    logging.info(f"Finished parallel computing in {comp_time} s.")
    logging.info("Saving Results")

    # Save all the results and get a dictionary of image classifications
    results_dict, results_df = classifier.save_all(results)



    # Assess accuracy
    logging.info("Calculating accuracy")
    correct_pred, false_pred = 0, 0
    total_s, total_w, total_n = 0, 0, 0
    correct_s, correct_w, correct_n = 0,0,0

    for i in range(len(p_list)) :
        patient_id = p_list[i]
        label = int(l_list[i])
        pred = int(results_df['prediction'].loc[i])

        logging.info(f"{patient_id}: pred={pred}, label={label}")

        # pred is either 0 or 2. label is either 0, 1, or 2
        # if pred==2 and label==1 we consider this a correct classification
        if abs(pred - label) < 1.9 :
            correct_pred = correct_pred + 1
        else :
            false_pred =  false_pred + 1

        # Get number of correct predictions for all classes
        if label == 1 :
            total_w = total_w + 1
            if pred == 2 :
                correct_w = correct_w + 1
        elif label == 0 :
            total_n = total_n + 1
            if pred == 0 :
                correct_n = correct_n + 1
        elif label == 2 :
            if pred == 2 :
                correct_s = correct_s + 1




    total_s = len(p_list) - total_n - total_w




    logging.info(f"{correct_pred} predictions were correct.\n{false_pred} predictions were incorrect.")

    logging.info(f"Data Summary:\nImages with: strong artifacts: {total_s}, weak: {total_w}, none: {total_n}")
    logging.info(f"correct strong artifact prediction rate: {correct_s/total_s}")
    logging.info(f"Proportion of weak artifacts that were detected: {correct_w/total_w}")
    logging.info(f"False positive rate: {(total_n - correct_n)/total_n}")
