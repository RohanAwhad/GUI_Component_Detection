#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 4            # number of cores 
#SBATCH -t 0-02:00:00   # time in d-hh:mm:ss
#SBATCH --mem=20G
#SBATCH -G 1
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required modules for job's environment
module load mamba/latest

# Using python, so source activate an appropriate environment
source activate interest_region_cls
echo $LD_LIBRARY_PATH
nvidia-smi

python -c "import torch; print(torch.cuda.is_available())"
python /home/rawhad/personal_jobs/GUI_Detection/GUI_Component_Detection/component_detection_trainer.py
