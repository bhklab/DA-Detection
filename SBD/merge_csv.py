import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser


csv1 = "/home/colin/Documents/BHKLab/data/reza_artifact_labels.csv"
csv2 = "/home/colin/Documents/BHKLab/data/artifact_labels_backup.csv"


def combine_two_df(df1, df2) :
    """Assumes the two DFs are the same shape and have same cols"""
    df1.dropna(inplace=True)
    df2.dropna(inplace=True)

    result = pd.concat([df1, df2])
    return result

def load_csvs(paths) :
    """paths is either a list of full paths to the CSVs to be combined,
    or a single path to a directory containing only the CSVs to be combined."""

    if type(paths) == list :
        # Load first DF
        df = pd.read_csv(paths[0], index_col="p_index",
                         dtype=str, na_values=['nan', 'NaN', ''])

        # Combine the other DFs with this one
        for file_path in paths[1:] :
            df_i = pd.read_csv(file_path, index_col="p_index",
                               dtype=str, na_values=['nan', 'NaN', ''])
            df = combine_two_df(df, df_i)
    else :
        # Get a list of full paths to the CSVs
        paths = [os.path.join(paths, i) for i in os.listdir(paths)]

        # Load first DF
        df = pd.read_csv(paths[0], index_col="p_index",
                         dtype=str, na_values=['nan', 'NaN', ''])

        # Combine the other DFs with this one
        for file_path in paths[1:] :
            df_i = pd.read_csv(file_path, index_col="p_index",
                               dtype=str, na_values=['nan', 'NaN', ''])
            df = combine_two_df(df, df_i)

    return df.sort_index()


# def main(args) :




if __name__ == "__main__" :

    parser = ArgumentParser()
    parser.add_argument("--base_path",
                        default="/cluster/home/carrowsm/logs/label/", type=str,
                        help="Path to a directory containing only the files to be combined.")

    parser.add_argument("--outfile", type=str, default="combined_labels.csv")


    args, file_list = parser.parse_known_args()

    if len(file_list) > 0 :
        file_list = [os.path.join(args.base_path, i) for i in file_list]
        df = load_csvs(file_list)

    else :
        df = load_csvs(args.base_path)


    df.to_csv(os.path.join(args.base_path, args.outfile))
