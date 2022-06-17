#!/bin/bash

####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=50000M
#SBATCH --partition=cpu2019
#SBATCH --mail-user=muhammadathar.ganaie@ucalgary.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/muhammadathar.ganaie/Brain-MR-Segmentation-Playground/preprocess_arc_output/output.out

####### Set environment variables ###############
export PYTHONPATH="."


####### Run your script ##########################
conda activate DA
ENV=PRODUCTION python data/preprocessing.py -data_dir /home/muhammadathar.ganaie/Dataset/cc359/philips3