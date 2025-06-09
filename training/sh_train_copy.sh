#!/bin/bash
#SBATCH --job-name=training_cont
#SBATCH --partition=ou_bcs_high
#SBATCH --output=../logs/training_%A_%a.out
#SBATCH --error=../logs/training_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1 
#SBATCH --mem=80G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wilke18@mit.edu
#SBATCH --exclude=node2802

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
echo "Running on task: $1"

conda activate thesis
python train_loop.py "sub-01" --name new_set_2_cont --batch_size 8 --task $1 --num_epochs 100 --patience 5 --delta .1 --num_trials 10 --parcellation $2

# # Only reach here if both scripts succeeded
# echo "Both scripts succeeded. Starting data transfer..."
# dest_dir="/orcd/data/satra/001/users/yibei/hcptrt/output/GLMsingle/results_v2/${TASK_ID}/"
# mkdir -p "$dest_dir"
# rsync -avz --ignore-existing /orcd/scratch/bcs/001/yibei/hcptrt/output/GLMsingle/results_v2/${TASK_ID}/ "$dest_dir"

# echo "All operations completed successfully!"
