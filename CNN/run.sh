#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --mem=200G
#SBATCH -J MIRA_CNN
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gres=gpu:1

echo 'Starting Shell Script'

source /cluster/home/carrowsm/.bashrc
conda activate radcure

path=/cluster/home/carrowsm/MIRA/run_on_radcure.py


echo 'Started python script.'
python $path
echo 'Python script finished.'
