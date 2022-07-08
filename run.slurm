#!/bin/bash

####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=100000
#SBATCH --mail-user=muhammadyusuf.hassan@ucalgary.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/muhammadyusuf.hassan/Brain-MR-Segmentation-Playground/output_unlearn/output.out
#SBATCH --gres=gpu:v100:1

####### Set environment variables ###############
export PYTHONPATH="."


####### Run your script ##########################
conda activate DA
ENV=PRODUCTION python methods/runner.py /home/muhammadyusuf.hassan/Brain-MR-Segmentation-Playground/experiments/train_un_si3_adapt_ge3.json