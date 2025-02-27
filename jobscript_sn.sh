#!/bin/bash -l
#SBATCH --job-name=keras_tuning
#SBATCH --partition=qany
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=150:00:00
#SBATCH --output=logs/tuning_%j.out
#SBATCH --error=logs/tuning_%j.err

echo "========= Job started at $(date) =========="

# Load required software modules
ml purge                   # Clear all loaded modules
ml load anaconda3          # Load Anaconda (Python)
ml load gcc                # Load GCC (adjust if necessary)

# Activate your Python environment.
source activate ConnectEnvironment
cd $SLURM_SUBMIT_DIR

# Run the Python script.
python tuning/sn_hps.py

echo "========= Job finished at $(date) =========="