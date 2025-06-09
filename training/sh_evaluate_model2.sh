#!/bin/bash
#SBATCH --job-name=model_evaluation
#SBATCH --partition=ou_bcs_normal
#SBATCH --output=../logs/training_%A.out
#SBATCH --error=../logs/training_%A.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wilke18@mit.edu

source $HOME/.bashrc


# Run first script
#echo "Activating environment for first script..."
#mamba activate glmsingle
#if ! python hcptrt_GLMsingle_localizer_emotion.py "${TASK_ID}"; then
#    echo "First script failed! Exiting..."
#    mamba deactivate
#    exit 1
#fi
#mamba deactivate

conda activate thesis
python evaluate_model2.py --parcellation "$1" --name "$2" --binary $3

# # Only reach here if both scripts succeeded
# echo "Both scripts succeeded. Starting data transfer..."
# dest_dir="/orcd/data/satra/001/users/yibei/hcptrt/output/GLMsingle/results_v2/${TASK_ID}/"
# mkdir -p "$dest_dir"
# rsync -avz --ignore-existing /orcd/scratch/bcs/001/yibei/hcptrt/output/GLMsingle/results_v2/${TASK_ID}/ "$dest_dir"

# echo "All operations completed successfully!"
