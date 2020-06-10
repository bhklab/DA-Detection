import numpy as np
import pandas as pd
import json
import os

"""Script which assesses the accuracy of the classifier on a per-slice basis"""


def load_results(results_file) :
    """ Load a JSON which contains each patient
    and a list of slices which contain artifacts
    for that patient"""
    with open(results_file) as json_file :
        dict = json.load(json_file)
    return dict

def load_labels(label_file) :
    """ Load a CSV containing each patient and
    the slice index which contains artifacts"""
    df = pd.read_csv(label_file, index_col="p_index",
                     dtype=str, na_values=['nan', 'NaN', '']).dropna()

    return df

def per_slice_acc(labels_df, preds_dict, offset=80) :
    # Loop through all classified patients and caluclate if their
    # slice prediction was correct.
    correct_preds = []                     # List of correctly classified patient IDs

    print("patient ID, Has Artifact, true_slice, pred_slice")

    for i in labels_df.index :
        patient_id = labels_df.at[i, "patient_id"]
        label = int(labels_df.at[i, "a_slice"])   # Slice index of strongest artifact
        preds = preds_dict[str(patient_id)]       # Predicted slice index of artifact(s)

        # Go through all predicted labels and see if they're right
        # If a given slice prediction is within +/- 5 of the truth, call it correct.
        print(patient_id, labels_df.at[i, "has_artifact"], label, [int(i)+80 for i in preds])

        for slice_ind in preds :
            if label-10 < (int(slice_ind) + offset) < label+10 :
                if len(correct_preds) == 0 :
                    """Add the correctly predicted index to the list
                    only do this if the last values wasn't from this
                    patient (to avoid double counting)"""
                    correct_preds.append(patient_id)
                else:
                    if correct_preds[-1] != patient_id :
                        correct_preds.append(patient_id)



            else :
                print(patient_id)
                continue
    return correct_preds




def main() :
    results_path = "/cluster/home/carrowsm/logs/label/preds.json"
    label_path = "/cluster/home/carrowsm/logs/label/reza_artifact_labels.csv"


    results_dict = load_results(results_path)
    labels_df  = load_labels(label_path)

    correct_preds = per_slice_acc(labels_df, results_dict)

    # How many of the identified artifacts got the slice right?
    r = len(correct_preds) / len(results_dict)
    print(f"Of the correctly predicted patients, {r*100} % had correctly predicted slice indeces.")




if __name__ == "__main__" :
    results_path = "/cluster/home/carrowsm/logs/label/preds.json"
    label_path = "/cluster/home/carrowsm/logs/label/reza_artifact_labels.csv"


    results_dict = load_results(results_path)
    labels_df  = load_labels(label_path)

    correct_preds = per_slice_acc(labels_df, results_dict)

    nb_preds = [l[0] for l in results_dict.values() if l]

    # How many of the identified artifacts got the slice right?
    r = len(correct_preds) / len(nb_preds)
    print(f"Of the correctly predicted patients, {r*100} % had correctly predicted slice indeces.")
