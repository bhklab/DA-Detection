#!/bin/bash
#SBATCH -t 0-24:00:00
#SBATCH --mem=25G
#SBATCH -J DA_Classification
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gres=gpu:1


echo 'Starting Shell Script'

source /cluster/home/carrowsm/.bashrc
conda activate radcure

script="classify_images.py"

echo 'Started python script.'
python $script --cnn_only --on_gpu
echo 'Python script finished.'
