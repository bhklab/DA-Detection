from argparse import ArgumentParser


### EDIT THESE PATHS ###
# img_path = "/cluster/projects/bhklab/RADCURE/img/"
img_path = "/cluster/projects/radiomics/Temp/RADCURE-npy/img"
img_suffix = "_img.npy"                 # This string follows the patient ID in the filename
label_path = "/cluster/home/carrowsm/data/radcure_DA_labels.csv"
log_dir = "/cluster/home/carrowsm/logs/label/"
### ---------------- ###

parser = ArgumentParser()
parser.add_argument("--img_dir", default=img_path, type=str)
parser.add_argument("--img_suffix", default=img_suffix, type=str)
parser.add_argument("--calc_acc", action='store_true',
                    help='Whether or not to calculate the accuracy of predictions, based on image labels.')
parser.add_argument("--label_dir", default=label_path, type=str, help='Path to a CSV containing image labels.')
parser.add_argument("--logging", action='store_true', help='Whether or not to save results.')
parser.add_argument("--logdir", default=log_dir, type=str, help='Where to save results.')


parser.add_argument("--norm_lower", default=0.64, type=float)
parser.add_argument("--norm_upper", default=0.86, type=float)

parser.add_argument("--test", action='store_true', help="If the test option is given, code will only process a few images.")

parser.add_argument("--ncpu", default=None, type=int, help="Number of CPUs to use.")

def get_args():
    args, unparsed = parser.parse_known_args()

    return args, unparsed
