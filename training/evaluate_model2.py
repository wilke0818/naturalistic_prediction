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
import torch.optim as optim

from tqdm import tqdm

import math

from torch.utils.data import WeightedRandomSampler
import csv
import json

import optuna

# At the top of the file, after imports
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


"""
def get_subject_localizer_data(sub_id, task, localizer_output_dir, parcellation='4s1056'):
    # sub-01_wm_stats.npz
    run_dir = os.path.join(localizer_output_dir,sub_id,task)
    #runs = [f for f in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir,f))]
    
    localizers = []
    
    localizer_results = np.load(os.path.join(run_dir,run,f'{sub_id}_{task}_stats_{parcellation}.npz'),allow_pickle=True)
    if binary:
    localizer_data = localizer_results['p_values']
    localizers.append({
        'data': localizer_data,
        'task': task,
        'subject': sub_id
    })
    return localizers
"""

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




#Need the arguments for each model
#If we have the held out episodes
#  need to load each episode
#  need to load each subject's data for the episode
#Need to load each subject's all localizer
#Need to possibly load other localizers for each subject to compare against
#average_mask.npy - within a subject averaging
#average_averaged_mask.npy - held out subject
#glm_single_averaged_mask.npy - held out subject

#Model directory
#/orcd/scratch/bcs/001/wilke18/friends/test3/simple_plis_model/emotion/
#sub-01_best.pt
#sub-01_validation_and_test_set.csv

#Output:
# For an experiment:
#     For each localizer:
#         For each held-out-subjects results:
#             Average the correlation of model runs with glmsingle
#             Check correlations between model runs
#             Check correlation between model runs and other methods of generating masks
#         Do the above for trained on individuals
#         Average the averages for a composite score

def dice_auc(y_true, y_pred):
    thresholds = [i/100 for i in range(1,100)]
    results = []
    for thresh in thresholds:
        y_pred_binary = np.zeros(y_pred.shape[0])
        idxs = np.where(y_pred > thresh)[0]
        y_pred_binary[idxs] = 1
        results.append(1-dice(y_true, y_pred_binary))
    results = [1-dice(y_true,np.ones(y_pred.shape[0]))] + results + [1-dice(y_true,np.zeros(y_pred.shape[0]))]
    auc = 0
    for i in range(1,len(results)):
        auc+=.01*(results[i]+results[i-1])/2
    return auc


def dice_at_thresh(y_true, y_pred, thresh=.5):
    y_pred_binary = np.zeros(y_pred.shape[0])
    idxs = np.where(y_pred > thresh)[0]
    y_pred_binary[idxs] = 1
    result = 1-dice(y_true, y_pred_binary)
    return result


def pr_auc(y_true, y_pred):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    return metrics.auc(recall, precision)

device = 'cuda'

def evaluate_with_held_out_subject(
        held_out_subject, 
        task, model_dir, 
        localizer_dir, 
        friends_scan_data_dir, 
        friends_stimuli_dir,
        parcellation,
        binary,
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
        localizer_data = get_subject_localizer_data(subject, task, localizer_dir, parcellation,binary)
        subject_ground_truths[subject] = localizer_data[0]['data']
        stimuli_data = [model_class._get_stimuli_embeddings(friends_stimuli_dir, friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_brain_data]
        friends_episode_metadata = [(friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_brain_data]
        sub_dataset = SubjectTrialDataset(subject,[data['data'] for data in friends_brain_data], [data['data'] for data in localizer_data], stimuli_data, friends_episode_metadata, train_method='aggregate')
        sub_datasets[subject] = sub_dataset

    friends_held_out_test_data = get_subject_friends_data(held_out_subject, friends_scan_data_dir,test_set)
    localizer_data = get_subject_localizer_data(held_out_subject, task, localizer_dir, parcellation,binary)
    held_out_localizer = localizer_data[0]['data']
    stimuli_held_out_test_data = [model_class._get_stimuli_embeddings(friends_stimuli_dir, friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_held_out_test_data]
    friends_held_out_test_episode_metadata = [(friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_held_out_test_data]

    held_out_test_dataset = SubjectTrialDataset(held_out_subject,[data['data'] for data in friends_held_out_test_data], [data['data'] for data in localizer_data], stimuli_held_out_test_data, friends_held_out_test_episode_metadata, train_method='aggregate')

    test_out_dist_loader = DataLoader(held_out_test_dataset, shuffle=False, collate_fn=collate_fn)

    best_ckpt = torch.load(f"{model_dir}/{held_out_subject}_best.pt", map_location=device)
    final_model = model_class(**best_ckpt['config']).to(device)
    final_model.load_state_dict(best_ckpt['model_state_dict'])
    final_model.eval()

   
    subject_results = {}
    subject_maps = {}
    # first get results for binary or continuous - dice, r2, roc-auc; dice will need some auc metric?
    for subject in sub_datasets:
        subject_results[subject] = {}
        subject_maps[subject] = {}
        trained_loader = DataLoader(sub_datasets[subject],shuffle=False,collate_fn=collate_fn)
        subject_result = {}
        for batch in trained_loader:
            fmri = batch['fmri'].to(device)
            stimulus = batch['stimulus'].to(device)
            targets = batch['localizer'].to(device)
            lengths = batch['lengths'].to(device)
            metadata = batch['metadata']
            outputs = final_model(fmri, stimulus, lengths)

            outputs = outputs.detach().cpu()
            targets = targets.detach().cpu()
            for i in range(targets.shape[0]):
                if binary:
                    outputs[i] = torch.sigmoid(outputs[i])
                    roc_auc = metrics.roc_auc_score(targets[i], outputs[i])
                    dice_score = dice_auc(targets[i],outputs[i])
                    pr_auc_score = pr_auc(targets[i],outputs[i])
                    dice30 = dice_at_thresh(targets[i],outputs[i], .3)
                    dice50 = dice_at_thresh(targets[i],outputs[i], .5)
                    dice70 = dice_at_thresh(targets[i],outputs[i], .7)
                    if 'dice' not in subject_result:
                        subject_result['dice'] = []
                        subject_result['dice-auc'] = []
                        subject_result['dice30'] = []
                        subject_result['dice70'] = []
                    if 'roc-auc' not in subject_result:
                        subject_result['roc-auc'] = []
                    if 'pr-auc' not in subject_result:
                        subject_result['pr-auc'] = []
                    ep_id = f"s{metadata[i][0]}e{metadata[i][1]}{metadata[i][2]}"
                    subject_result['dice-auc'].append(dice_score)
                    subject_result['roc-auc'].append(roc_auc)
                    subject_result['pr-auc'].append(pr_auc_score)
                    subject_result['dice'].append(dice50)
                    subject_result['dice30'].append(dice30)
                    subject_result['dice70'].append(dice70)
                    subject_maps[subject][ep_id] = outputs[i]
                else:
                    loss = metrics.r2_score(targets[i], outputs[i])
                    if 'r2' not in subject_result:
                        subject_result['r2'] = []
                    ep_id = f"s{metadata[i][0]}e{metadata[i][1]}{metadata[i][2]}"
                    subject_result['r2'].append(loss)
                    subject_maps[subject][ep_id] = outputs[i]
        if binary:
            subject_results[subject]['dice'] = np.average(subject_result['dice'])
            subject_results[subject]['roc-auc'] = np.average(subject_result['roc-auc'])
            subject_results[subject]['pr-auc'] = np.average(subject_result['pr-auc'])
            subject_results[subject]['dice-auc'] = np.average(subject_result['dice-auc'])
            subject_results[subject]['dice30'] = np.average(subject_result['dice30'])
            subject_results[subject]['dice70'] = np.average(subject_result['dice70'])
        else:
            subject_results[subject]['r2'] = np.average(subject_result['r2'])


    subject_results[held_out_subject] = {}
    subject_maps[held_out_subject] = {}
    subject_result = {}
    for batch in test_out_dist_loader:
            fmri = batch['fmri'].to(device)
            stimulus = batch['stimulus'].to(device)
            targets = batch['localizer'].to(device)
            lengths = batch['lengths'].to(device)
            metadata = batch['metadata']
            outputs = final_model(fmri, stimulus, lengths)

            outputs = outputs.detach().cpu()
            targets = targets.detach().cpu()
            for i in range(targets.shape[0]):
                if binary:
                    outputs[i] = torch.sigmoid(outputs[i])
                    roc_auc = metrics.roc_auc_score(targets[i], outputs[i])
                    dice_score = dice_auc(targets[i],outputs[i])
                    pr_auc_score = pr_auc(targets[i],outputs[i])
                    dice30 = dice_at_thresh(targets[i],outputs[i], .3)
                    dice50 = dice_at_thresh(targets[i],outputs[i], .5)
                    dice70 = dice_at_thresh(targets[i],outputs[i], .7)
                    if 'dice' not in subject_result:
                        subject_result['dice'] = []
                        subject_result['dice-auc'] = []
                        subject_result['dice30'] = []
                        subject_result['dice70'] = []
                    if 'roc-auc' not in subject_result:
                        subject_result['roc-auc'] = []
                    if 'pr-auc' not in subject_result:
                        subject_result['pr-auc'] = []
                    ep_id = f"s{metadata[i][0]}e{metadata[i][1]}{metadata[i][2]}"
                    subject_result['dice-auc'].append(dice_score)
                    subject_result['roc-auc'].append(roc_auc)
                    subject_result['pr-auc'].append(pr_auc_score)
                    subject_result['dice'].append(dice50)
                    subject_result['dice30'].append(dice30)
                    subject_result['dice70'].append(dice70)
                    subject_maps[held_out_subject][ep_id] = outputs[i]
                else:
                    loss = metrics.r2_score(targets[i], outputs[i])
                    if 'r2' not in subject_result:
                        subject_result['r2'] = []
                    ep_id = f"s{metadata[i][0]}e{metadata[i][1]}{metadata[i][2]}"
                    subject_result['r2'].append(loss)
                    subject_maps[held_out_subject][ep_id] = outputs[i]
    if binary:
        subject_results[held_out_subject]['dice'] = np.average(subject_result['dice'])
        subject_results[held_out_subject]['roc-auc'] = np.average(subject_result['roc-auc'])
        subject_results[held_out_subject]['pr-auc'] = np.average(subject_result['pr-auc'])
        subject_results[held_out_subject]['dice-auc'] = np.average(subject_result['dice-auc'])
        subject_results[held_out_subject]['dice30'] = np.average(subject_result['dice30'])
        subject_results[held_out_subject]['dice70'] = np.average(subject_result['dice70'])
    else:
        subject_results[held_out_subject]['r2'] = np.average(subject_result['r2'])

    # then check whether for the same subject we're always getting the same maps out; check relationship between subject's maps

    subject_map_consistency = {}
    for subject in subject_maps:
        consistency = []
        for i in range(len(subject_maps[subject])):
            eps = list(subject_maps[subject].keys())
            subject_map_i = subject_maps[subject][eps[i]]
            for j in range(i+1,len(subject_maps[subject])):
                subject_map_j = subject_maps[subject][eps[j]]
                consistency.append(np.corrcoef(subject_map_i,subject_map_j))
        subject_map_consistency[subject] = np.average(consistency)


    # then check held-out subject vs. average (regression to the average) vs. combination methods?
    held_out_consistency_in_model = {}
    held_out_consistency_ground_truth = {}
    average_episode_map = {}
    average_localizer_map = []
    for subject in other_subjects:
        average_localizer_map.append(subject_ground_truths[subject])
        consistency_in_model = []
        for ep_id in subject_maps[held_out_subject]:
            if ep_id not in average_episode_map:
                average_episode_map[ep_id] = []
            average_episode_map[ep_id].append(subject_maps[subject][ep_id])
            consistency_in_model.append(np.corrcoef(subject_maps[held_out_subject][ep_id],subject_maps[subject][ep_id]))
        held_out_consistency_in_model[subject] = np.average(consistency_in_model)

    # does the average of subjects for each episodes coincide with held out
    average_episode = []
    average_ground_truth_w_episodes = []
    for ep_id in subject_maps[held_out_subject]:
        average_ground_truth_w_episodes.append(np.corrcoef(subject_maps[held_out_subject][ep_id],np.average(average_localizer_map,axis=0)))
        average_episode.append(np.corrcoef(subject_maps[held_out_subject][ep_id],np.average(average_episode_map[ep_id],axis=0)))
    corr_of_average_episode = np.average(average_episode)
    corr_of_average_ground_truths_w_ground_truth = np.corrcoef(held_out_localizer,np.average(average_localizer_map,axis=0))[0][1]
    corr_of_average_ground_truth_w_episodes = np.average(average_ground_truth_w_episodes)

    examples_dir = f"{model_dir}/{held_out_subject}_brain_maps/"
    for i in range(5):
        ep_id = list(subject_maps[held_out_subject].keys())[i]
        for subject in subject_maps:
            subject_example_dir = os.path.join(examples_dir, subject)
            os.makedirs(subject_example_dir, exist_ok=True)
            
            np.save(f"{subject_example_dir}/{ep_id}.npy", subject_maps[subject][ep_id])


    return {
            'metric_results': subject_results,
            'corr_of_average_episode': corr_of_average_episode,
            'corr_of_average_ground_truths_w_ground_truth': corr_of_average_ground_truths_w_ground_truth,
            'corr_of_average_ground_truth_w_episodes': corr_of_average_ground_truth_w_episodes,
            'held_out_consistency_in_model': held_out_consistency_in_model,
            'subject_map_consistency': subject_map_consistency
    }


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
    parser.add_argument("--binary", type=int, default=0)
    args = parser.parse_args()

    localizer_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/mask/{args.parcellation}"
    model_dir = f"{checkpoint_base_dir}/{args.parcellation}/{args.name}/"
    
    friends_scan_data_dir = os.path.join(friends_scan_data_dir,args.parcellation)

    model_types = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir,f))]
    for model_type in model_types:
        model_type_dir = os.path.join(model_dir, model_type)
        model_class = model_testing_classes[model_type]
        tasks = [f for f in os.listdir(model_type_dir) if os.path.isdir(os.path.join(model_type_dir,f))]
        subjects = [f'sub-0{i}' for i in range(1,7)]
        results = {}
        for task in tasks:
            print('Running task', task)
            results[task] = {}
            task_dir = os.path.join(model_type_dir,task)
            for held_out_subject in subjects:
                print('\t on', held_out_subject)
                results[task][held_out_subject] = evaluate_with_held_out_subject(held_out_subject, task, task_dir, localizer_dir, friends_scan_data_dir, friends_stimuli_dir, args.parcellation,bool(args.binary),model_class)
                #print(results[task][held_out_subject])

        with open(os.path.join(model_type_dir,f"{args.parcellation}_{args.name}_{model_class.MODEL_NAME}_results.json"),"w") as f:
            json.dump(results, f)


if __name__=="__main__":
    main()
