#!/bin/bash -l
#SBATCH --job-name=keras_tuning      # Job name
#SBATCH --partition=qgpu              # GPU partition (change to q64 for CPU)
#SBATCH --gres=gpu:1                 # Request 1 GPU (remove this line for CPU-only)
#SBATCH --mem=32G                    # Request 32GB memory (adjust as needed)
#SBATCH --cpus-per-task=8            # Request 8 CPU cores
#SBATCH --time=12:00:00              # Max job runtime (12 hours)
#SBATCH --output=logs/tuning_%j.out  # Standard output log
#SBATCH --error=logs/tuning_%j.err   # Error log

echo "========= Job started at $(date) =========="

# Load required software modules
ml purge                   # Clear all loaded modules
ml load anaconda3          # Load Anaconda (Python)
ml load cuda/12.0.0        # Load CUDA (adjust if necessary)
ml load gcc                # Load GCC (adjust if necessary)

# Activate your Conda environment (replace "my_env" with your actual environment name)
source activate ConnectEnvironment

# Move to job submission directory
cd $SLURM_SUBMIT_DIR

# Run the Python script
python tuning/dg_hyperparameter_search.py

echo "========= Job finished at $(date) =========="
