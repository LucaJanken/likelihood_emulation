#!/bin/bash -l
#SBATCH --job-name=keras_tuning      # Job name
#SBATCH --partition=qgpu             # GPU partition (change to q64 for CPU)
#SBATCH --gres=gpu:1                 # Request 1 GPU (remove this line for CPU-only)
#SBATCH --mem=64G                    # Request 32GB memory (adjust as needed)
#SBATCH --cpus-per-task=32            # Request 8 CPU cores
#SBATCH --time=150:00:00              # Max job runtime (12 hours)
#SBATCH --output=logs/tuning_%j.out  # Standard output log
#SBATCH --error=logs/tuning_%j.err   # Error log

echo "========= Job started at $(date) =========="

# Load required software modules
ml purge                   # Clear all loaded modules
ml load anaconda3          # Load Anaconda (Python)
ml load cuda/12.0.0        # Load CUDA (adjust if necessary)
ml load gcc                # Load GCC (adjust if necessary)

# Set cuDNN paths (from MATLAB R2022b installation)
export LD_LIBRARY_PATH=/home/comm/swstack/core/matlab/R2022b/bin/glnxa64:$LD_LIBRARY_PATH
export CPATH=/home/comm/swstack/core/matlab/R2022b/bin/glnxa64:$CPATH
export LIBRARY_PATH=/home/comm/swstack/core/matlab/R2022b/bin/glnxa64:$LIBRARY_PATH

# Print paths for debugging
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CPATH: $CPATH"
echo "LIBRARY_PATH: $LIBRARY_PATH"

# Activate your Conda environment (replace "ConnectEnvironment" with your actual environment name)
source activate ConnectEnvironment

# Move to job submission directory
cd $SLURM_SUBMIT_DIR

# Check GPU availability before running the script
python -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')))"

# Run the Python script
python tuning/chain_6D_hps.py

echo "========= Job finished at $(date) =========="
