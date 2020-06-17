#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --mem=25G
#SBATCH -J DA_Classification
#SBATCH -c 35
#SBATCH -N 1
#SBATCH --partition=all


echo 'Starting Shell Script'

source /cluster/home/carrowsm/.bashrc
conda activate radcure

script="classify_images.py"

echo 'Started python script.'
python $script --sbd_only --ncpu=75
echo 'Python script finished.'
