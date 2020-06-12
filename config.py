from argparse import ArgumentParser


### EDIT THESE PATHS ###
img_path = "/cluster/projects/radiomics/Temp/RADCURE-npy/img"
label_path = "/cluster/home/carrowsm/data/radcure_DA_labels.csv"
log_dir = "/cluster/home/carrowsm/logs/label/"
### ---------------- ###

parser = ArgumentParser()


### General I/O options ###
parser.add_argument("--img_dir", default=img_path, type=str)
parser.add_argument("--img_suffix", default="", type=str,
                    help="A string which occurs after the patient ID, before the file extension.")
parser.add_argument("--label_dir", default=label_path, type=str, help='Path to a CSV containing image labels.')
parser.add_argument("--logging", action='store_true', help='Whether or not to save results.')
parser.add_argument("--out_path", default=log_dir, type=str, help='Where to save results.')


### Arguments for both models ###
parser.add_argument("--test", action='store_true', help="If the test option is given, code will only process a few images.")


### Arguments specific to SBD ###
parser.add_argument("--ncpu", default=None, type=int, help="Number of CPUs to use.")


### Arguments specific to CNN ###
parser.add_argument("--on_gpu", action='store_true', help="Number of CPUs to use.")


args, unparsed = parser.parse_known_args()


def get_args():
    args, unparsed = parser.parse_known_args()

    return args, unparsed
