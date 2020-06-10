#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --mem=20G
#SBATCH -J ArtifactClassifier
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
##  --#SBATCH --account=radiomics
#SBATCH --partition=all

# This script runs the DA location finding algorithm
# located in find_loc.py on a slurm cluster

echo 'Starting Shell Script'

source /cluster/home/carrowsm/.bashrc
conda activate radcure

# Arguments to pass to classifier
path=/cluster/home/carrowsm/artifacts/location_finder/find_loc.py

echo 'Started python script.'
python $path
echo 'Python script finished.'
