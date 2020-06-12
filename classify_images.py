"""
This is the main script in the DA-Detection package which classifies a set of CT images
as containing strong, weak, or no dental artifacts (DA).
"""
import os
import numpy as np
import pandas as pd
import multiprocessing
import torch


from data_loading.data_loader import DataLoader
from SBD.classify import Classifier
from CNN.run_on_radcure import classify_img
from CNN import DAClassification
from config import get_args



def setup_SBD(args, data_loader) :
    """ This function handles all setup for sinogram-based DA detection (SBD).
    It takes the path to the DA labels, the path to the directory containing
    .DICOM files (each )
    """
    classifier = Classifier(args, data_loader)

    # Setup parallel processes
    # Get number of available CPUs
    if num_cpus == None :
        num_cpus = multiprocessing.cpu_count()
    else :
        num_cpus = args.ncpu

    # Make a pool of parallel workers
    pool = multiprocessing.Pool(num_cpus)

    # Create list of tasks. (Matches indices in data_loader)
    tasks = np.arange(0, len(data_loader), 1)
    return pool, tasks, classifier



def setup_CNN(args, data_loader) :
    # GPU check
    if args.on_gpu and torch.cuda.is_available() :
        print("Running CNN on GPU")
        on_gpu = True
    else :
        print("Runnnig CNN on CPU")
        on_gpu = False


    # NETWORK LOADING #
    # net_path = os.path.abspath(os.path.dirname(sys.argv[0]))
    net_path = "/cluster/home/carrowsm/artifacts/DA-Detection/CNN/"
    net_name = 'testCheckpoint.pth.tar'

    network = DAClassification.LoadNet(net_path, net_name, on_gpu=True)

    # GPU Computing
    torch.cuda.set_device(0)    # Set the device to a GPU
    network.cuda()              # Put the model on the GPU
    #-###############-#

    return network, on_gpu




def run_SBD(sbd_pool, sbd_tasks, sbd_classifier) :
    with pool as p :
        sbd_results = p.map(sbd_classifier.classify, tasks)

    # Each value in sbd_results is a tuple with
    # (patient index, patient ID, binaryprediction, location prediction indeces)
    # Save the results and get a data frame of the binary predictions and
    # a dictionary of the location predictions (for predicted DA+ imgs only)
    sbd_loc_dict, sbd_class_df = classifier.save_all(sbd_results)



def run_cnn(network, on_gpu, data_loader, out_path) :
    # Classify each image with CNN
    preds = []
    prob1 = []
    prob2 = []

    for path in data_loader.patient_list :
        # Get full path to DICOM
        img_path = data_loader.get_dicom_path(path)

        # Forward pass though model
        predicted_label, softmax_prob = classify_img(img_path, net=network, on_gpu=on_gpu)

        preds.append(predicted_label.item())
        prob1.append(softmax_prob[0].item())
        prob2.append(softmax_prob[1].item())

    # Create and save a CSV of results
    path = os.path.join(out_path, "CNN/binary_class.csv")
    df = pd.DataFrame(data={"p_index": np.arange(0, len(data_loader)),
                            "patient_id": data_loader.patient_list,
                            "CNN_preds": preds,
                            "CNN_probs0": prob1,
                            "CNN_probs1": prob2}, dtype=str)
    df = df.set_index("p_index")
    df.to_csv(path)



def decision_network(sbd_predictions, cnn_predictions) :
    pass

def main(args, csv_path, img_path, out_path) :
    # Initialize data loader
    data_loader = DataLoader(img_path, csv_path, img_suffix="", file_type="dicom")

    # Flag to run both classifiers
    both = args.sbd_only is False and args.cnn_only is False

    if args.sbd_only or both:
        print("Running SBD")
        # ### SINOGRAM-BASED DETECTION ###
        sbd_pool, sbd_tasks, sbd_classifier = setup_SBD(args, data_loader)
        run_SBD(sbd_pool, sbd_tasks, sbd_classifier)
        # ### ------------------------ ###

    if args.cnn_only or both :
        print("Running CNN")
        # ### -- CNN-BASED DETECTION - ###
        network, on_gpu = setup_CNN(args, data_loader)
        run_cnn(network, on_gpu, data_loader, out_path)
        # ### ------------------------ ###

    if args.cnn_only and args.sbd_only :
        raise Exception("Use only one of '--sbd_only' or '--cnn_only'")



if __name__ == '__main__':

    args, unparsed = get_args()
    print(args)

    csv_path = args.label_dir
    img_path = args.img_dir
    out_path = args.out_path

    main(args, csv_path, img_path, out_path)
