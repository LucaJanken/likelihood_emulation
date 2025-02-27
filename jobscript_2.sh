#!/bin/bash -l
#SBATCH --job-name=keras_tuning_mpi
#SBATCH --partition=qany
#SBATCH --nodes=4                      # Total of 5 nodes.
#SBATCH --ntasks-per-node=1            # One MPI process per node.
#SBATCH --cpus-per-task=24             # Allocate 40 CPU cores per process.
#SBATCH --mem=128G                      # Memory per node (adjust as needed).
#SBATCH --time=150:00:00
#SBATCH --output=logs/tuning_%j.out
#SBATCH --error=logs/tuning_%j.err

echo "========= Job started at $(date) =========="

ml purge
ml load anaconda3
ml load gcc

# Set threading so TensorFlow uses the allocated cores.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate your Python environment.
source activate ConnectEnvironment
cd $SLURM_SUBMIT_DIR

# Launch the MPI-enabled Python script with srun.
srun python tuning/chain_6D_hps_MPI.py

echo "========= Job finished at $(date) =========="
