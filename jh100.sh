#!/bin/bash
#SBATCH --job-name=my_job2a        # Job name
#SBATCH --output=jobs/output_%j.txt   # Standard output and error log (%j expands to jobId)
#SBATCH --error=jobs/error_%j.txt     # Error file
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --time=19:00:00          # Run time (hh:mm:ss)
#SBATCH --gres=gpu:h100:1                 # Number of GPUs requested
#SBATCH --mem=20G                 # Memory requested (1 GB)
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your_email@example.com  # Where to send mail

# Load necessary modules (if any)
SINGULARITY_IMAGE="/home/sa7270/pytorch_latest.sif"

# Execute your command within the Singularity container
singularity exec --nv $SINGULARITY_IMAGE python run_experiment.py

# Your executab
