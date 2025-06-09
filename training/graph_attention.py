import pickle
import matplotlib.pyplot as plt

import os
import numpy as np
import math
import cv2
from moviepy import VideoFileClip
import argparse
from argparse import ArgumentParser
from dotenv import load_dotenv
from datetime import datetime
from subject_dataset import SubjectTrialDataset
from model_to_run import model_testing_classes
from scipy.spatial.distance import dice
from sklearn import metrics

import nibabel as nib
import os
import scipy.stats as st

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import logging
from torch.utils.data import DataLoader, ConcatDataset

from tqdm import tqdm

import math

import csv
import json

def setup_logging(base_dir, task, sub_id=None, parcellation='4s1056'):
    """Set up logging to both file and console."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(base_dir, 'logs', 'training')
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if sub_id:
        held_out = sub_id
    else:
        held_out = 'all'
    log_file = os.path.join(log_dir, f'{parcellation}_{task}_{held_out}_{timestamp}.log')

    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    logging.info(f"Logging to: {log_file}")

subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-05']

tmp_dir = 'jordan_features/tmp/'

def get_subject_friends_data(sub_id, friends_scan_path, subset=None):
    #example data sub-01_ses-011_task-s02e04a_space-fsLR_den-91k_bold_cleaned.dtseries.nii
    #print(friends_scan_path, sub_id)
    sub_data_path = os.path.join(friends_scan_path,sub_id)
    sub_scans = os.listdir(sub_data_path)
    data = []
    for scan in sub_scans:
        scan_details = scan.split('_')
        brain_data = np.load(os.path.join(sub_data_path,scan))
        #img = nib.load(os.path.join(sub_data_path,scan))
        #brain_data = img.get_fdata()
        if subset and (scan_details[2][6:8],scan_details[2][9:11],scan_details[2][11]) not in subset:
            continue
        data.append({
            'data': brain_data,#os.path.join(sub_data_path,scan),#torch.tensor(brain_data),
            'session': scan_details[1][4:7],
            'season': scan_details[2][6:8],
            'episode_num': scan_details[2][9:11],
            'episode_part': scan_details[2][11]
        })
    return data


def get_subject_localizer_data(sub_id, task, localizer_output_dir, parcellation, binary=False):
    # sub-01_wm_stats.npz
    threshold = 0.000000001
    run_dir = os.path.join(localizer_output_dir,sub_id,task)
    localizers = []
    
    localizer_results = np.load(os.path.join(run_dir,f'{sub_id}_{task}_stats_{parcellation}.npz'),allow_pickle=True)
    if binary:
        localizer_data = localizer_results['binary_mask']
    else:
        localizer_data = np.ones(localizer_results['p_values_corrected'].shape[0])
        idxs = np.where(localizer_results['t_stats'] > 0)[0]
        localizer_data[idxs] = localizer_results['p_values_corrected'][idxs]

        localizer_data = np.clip(localizer_data, a_min=threshold, a_max=1-threshold)
        localizer_data = st.norm.ppf(localizer_data)

    localizers.append({
        'data': localizer_data,
        'task': task,
        'subject': sub_id
    })
    return localizers







def extract_clip(video_file, tr_point, hrf_delay=4,tr_size=1.49, window_size=2):
    clip = VideoFileClip(video_file)
    start_tr = max(0,tr_point-hrf_delay)
    end_tr = start_tr + window_size
    start_time = start_tr*tr_size
    end_time = end_tr*tr_size
    video_segment = clip.subclipped(max(0,start_time), min(end_time,clip.duration))
    return video_segment, start_tr, end_tr


def extract_features_of_video(video_file, output_dir, features_func, tmp_file_name='current_file.mp4'):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open movie.")
        return
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = round(video_total_frames / video_fps, 2)
    cap.release()
    
    file_name = os.path.splitext(os.path.split(video_file)[1])[0]
    number_of_trs = math.ceil(video_duration/1.49)
    print('trs',number_of_trs)
    for tr in range(4,number_of_trs):
        output_file_path = os.path.join(output_dir, f"{file_name}_{tr}.npy")
        if os.path.exists(output_file_path):
            continue
        clip, _, _ = extract_clip(video_file, tr)
        tmp_file_path = os.path.join(tmp_dir, tmp_file_name)
        clip.write_videofile(tmp_file_path, codec="libx264", audio_codec="aac")
        #output_file_path = os.path.join(output_dir, f"{file_name}_{tr}.npy")
        features_func(tmp_file_path, output_file_path)

def get_episode_from_metadata(metadata):
    season, episode_num, _ = metadata
    numerical_episode = 0
    if season=='01':
        numerical_episode += int(episode_num)
        return numerical_episode
    else:
        numerical_episode += 24

    if season=='02':
        numerical_episode += int(episode_num)
        return numerical_episode
    else:
        numerical_episode += 24

    if season=='03':
        numerical_episode += int(episode_num)
        return numerical_episode
    else:
        numerical_episode += 25

    if season=='04':
        numerical_episode += int(episode_num)
        return numerical_episode
    else:
        numerical_episode += 24

    if season=='05':
        numerical_episode += int(episode_num)
        return numerical_episode
    else:
        numerical_episode += 24

    assert season=='06'
    numerical_episode += int(episode_num)
    return numerical_episode

def collate_fn(batch):
    fmri_seqs = [item["fmri"] for item in batch]
    stim_ids = torch.stack([item["stimulus"] for item in batch])
    targets = torch.stack([item["localizer"] for item in batch])
    metadatas = [item['metadata'] for item in batch]

    # Pad sequences to max length in batch
    padded_seqs = torch.nn.utils.rnn.pad_sequence(fmri_seqs, batch_first=True)  # shape: (batch, max_seq_len, 91248)
    lengths = torch.tensor([seq.shape[0] for seq in fmri_seqs])
    #print(metadatas)
    #raise ValueError('stop')
    return {
        "fmri": padded_seqs,
        "lengths": lengths,
        "stimulus": stim_ids,
        "localizer": targets,
        "metadata": metadatas
    }





device = 'cuda'

def evaluate_with_held_out_subject(
        held_out_subject, 
        task, model_dir, 
        localizer_dir, 
        friends_scan_data_dir, 
        friends_stimuli_dir,
        parcellation,
        model_class
        ):
    other_subjects = [f"sub-0{i}" for i in range(1,7) if f"sub-0{i}" !=  held_out_subject]
    test_set = []
    with open(f"{model_dir}/{held_out_subject}_validation_and_test_set.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['set'] == "test":
                test_set.append((row['season'], row['episode'], row['episode_part']))


    sub_datasets = {}
    subject_ground_truths = {}
    for subject in other_subjects:
        friends_brain_data = get_subject_friends_data(subject, friends_scan_data_dir,test_set)
        localizer_data = get_subject_localizer_data(subject, task, localizer_dir, parcellation,True)
        subject_ground_truths[subject] = localizer_data[0]['data']
        stimuli_data = [model_class._get_stimuli_embeddings(friends_stimuli_dir, friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_brain_data]
        friends_episode_metadata = [(friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_brain_data]
        sub_dataset = SubjectTrialDataset(subject,[data['data'] for data in friends_brain_data], [data['data'] for data in localizer_data], stimuli_data, friends_episode_metadata, train_method='aggregate')
        sub_datasets[subject] = sub_dataset

    friends_held_out_test_data = get_subject_friends_data(held_out_subject, friends_scan_data_dir,test_set)
    localizer_data = get_subject_localizer_data(held_out_subject, task, localizer_dir, parcellation,True)
    held_out_localizer = localizer_data[0]['data']
    stimuli_held_out_test_data = [model_class._get_stimuli_embeddings(friends_stimuli_dir, friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_held_out_test_data]
    friends_held_out_test_episode_metadata = [(friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_held_out_test_data]

    held_out_test_dataset = SubjectTrialDataset(held_out_subject,[data['data'] for data in friends_held_out_test_data], [data['data'] for data in localizer_data], stimuli_held_out_test_data, friends_held_out_test_episode_metadata, train_method='aggregate')

    test_out_dist_loader = DataLoader(held_out_test_dataset, shuffle=False, collate_fn=collate_fn)

    best_ckpt = torch.load(f"{model_dir}/{held_out_subject}_best.pt", map_location=device)
    final_model = model_class(**best_ckpt['config']).to(device)
    final_model.load_state_dict(best_ckpt['model_state_dict'])
    final_model.eval()

    output_dir = os.path.join(model_dir,'attention_graphs',held_out_subject)
    os.makedirs(output_dir, exist_ok=True)
    
    subject_attentions = {}
    for subject in sub_datasets:
        subject_attentions[subject] = {}
        trained_loader = DataLoader(sub_datasets[subject],shuffle=False,collate_fn=collate_fn)
        subject_result = {}
        for batch in trained_loader:
            fmri = batch['fmri'].to(device)
            stimulus = batch['stimulus'].to(device)
            targets = batch['localizer'].to(device)
            lengths = batch['lengths'].to(device)
            metadata = batch['metadata']
            outputs, attention = final_model(fmri, stimulus, lengths,output_temporal=True)

            outputs = outputs.detach().cpu()
            targets = targets.detach().cpu()
            attention = attention.detach().cpu()
            for i in range(targets.shape[0]):
                    ep_id = f"s{metadata[i][0]}e{metadata[i][1]}{metadata[i][2]}"
                    subject_attentions[subject][ep_id] = attention[i][5:-5]#outputs[i]
   
    subject_attentions[held_out_subject] = {}
    subject_result = {}
    for batch in test_out_dist_loader:
            fmri = batch['fmri'].to(device)
            stimulus = batch['stimulus'].to(device)
            targets = batch['localizer'].to(device)
            lengths = batch['lengths'].to(device)
            metadata = batch['metadata']
            outputs, attention = final_model(fmri, stimulus, lengths,output_temporal=True)

            outputs = outputs.detach().cpu()
            targets = targets.detach().cpu()
            attention = attention.detach().cpu()
            for i in range(targets.shape[0]):
                    ep_id = f"s{metadata[i][0]}e{metadata[i][1]}{metadata[i][2]}"
                    subject_attentions[held_out_subject][ep_id] = attention[i][5:-5]#outputs[i]

    for ep_id in subject_attentions[held_out_subject]:
        plt.figure(figsize=(12,8))
        for subj, subj_data in subject_attentions.items():
            y = subj_data[ep_id]
            x = list(range(len(y)))  # x-axis can be timepoints or indices
            if subj == held_out_subject:
                plt.plot(x, y, label=f"{subj} (held out)", linestyle='--', linewidth=2)
            else:
                plt.plot(x, y, label=subj, alpha=0.6)

        plt.title(f"Episode {ep_id}")
        plt.xlabel("Time")
        plt.ylabel("Attention Weight")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plot_{ep_id}_{task}_{held_out_subject}_{parcellation}_{model_class.MODEL_NAME}.png")
        plt.close()


def main():
    load_dotenv()
    friends_scan_data_dir = os.getenv("FRIENDS_SCAN_DATA_DIR")
    friends_stimuli_dir = os.getenv("FRIENDS_STIMULI_DIR")
    base_dir = os.getenv("FRIENDS_BASE_DIR")
    scratch_dir = os.getenv("FRIENDS_SCRATCH_DIR")
    checkpoint_base_dir = os.getenv("FRIENDS_CHECKPOINT_DIR")

    print('scan data dir', friends_scan_data_dir)
    print('stimuli dir', friends_stimuli_dir)
    print('base dir', base_dir)
    print('scratch dir', scratch_dir)
    print('checkpoint base dir', checkpoint_base_dir)

    parser = ArgumentParser(description="train model on fmri data")
    parser.add_argument("--name", type=str)
    parser.add_argument("--parcellation", type=str, default='4s1056')
    parser.add_argument("--model_name", type=str, default='mini_dice_model')
    args = parser.parse_args()

    localizer_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/mask/{args.parcellation}"
    model_dir = f"{checkpoint_base_dir}/{args.parcellation}/{args.name}/{args.model_name}/"
    
    friends_scan_data_dir = os.path.join(friends_scan_data_dir,args.parcellation)

    model_type_dir = model_dir#os.path.join(model_dir, model_type)
    model_class = model_testing_classes[args.model_name]
    tasks = [f for f in os.listdir(model_type_dir) if os.path.isdir(os.path.join(model_type_dir,f))]
    subjects = [f'sub-0{i}' for i in range(1,7)]
    results = {}
    for task in tasks:
        print('Running task', task)
        results[task] = {}
        task_dir = os.path.join(model_type_dir,task)
        for held_out_subject in subjects:
            print('\t on', held_out_subject)
            evaluate_with_held_out_subject(held_out_subject, task, task_dir, localizer_dir, friends_scan_data_dir, friends_stimuli_dir, args.parcellation,model_class)
            #print(results[task][held_out_subject])



if __name__=="__main__":
    main()
