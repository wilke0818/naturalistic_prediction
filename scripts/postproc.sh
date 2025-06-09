#!/bin/bash

#SBATCH --job-name=postproc_hcptrt
#SBATCH --partition=mit_normal
#SBATCH --output=../logs/postproc_hcptrt_%A_%a.out
#SBATCH --error=../logs/postproc_hcptrt_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-5
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wilke18@mit.edu

source ~/.bashrc
# Activate your Conda environment
conda activate thesis

sub_ids=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")

TASK_ID=${sub_ids[$SLURM_ARRAY_TASK_ID]}

echo "Processing: $TASK_ID"

python postproc.py "${TASK_ID}" "hcptrt" --global_zscore
