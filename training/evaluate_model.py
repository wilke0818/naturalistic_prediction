from argparse import ArgumentParser
from dotenv import load_dotenv
from datetime import datetime
from subject_dataset import SubjectTrialDataset
from model_to_run import model_testing_classes

import nibabel as nib
import os

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
def setup_logging(base_dir, task, sub_id=None):
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
    log_file = os.path.join(log_dir, f'{task}_{held_out}_{timestamp}.log')

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


def get_subject_localizer_data(sub_id, task, localizer_output_dir, glm_single_all=False):
    # sub-01_wm_stats.npz
    run_dir = os.path.join(localizer_output_dir,sub_id,task)
    runs = [f for f in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir,f))]
    if glm_single_all:
        runs = ['all']
    else:
        runs.remove('all')
    localizers = []
    for run in runs:
        localizer_results = np.load(os.path.join(run_dir,run,f'{sub_id}_{task}_stats_parcel.npz'),allow_pickle=True)
        localizer_data = localizer_results['p_values']
        localizers.append({
            'data': localizer_data,
            'task': task,
            'subject': sub_id,
            'run': run
        })
    return localizers


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


device = 'cuda'

def evaluate_with_held_out_subject(held_out_subject, task, model_dir, localizer_dir, friends_scan_data_dir, friends_stimuli_dir):
    other_subjects = [f"sub-0{i}" for i in range(1,7) if f"sub-0{i}" not in held_out_subject]
    test_set = []
    validation_set = []
    with open(f"{model_dir}/{held_out_subject}_validation_and_test_set.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['set'] == "test":
                test_set.append((row['season'], row['episode'], row['episode_part']))
            else:
                validation_set.append((row['season'], row['episode'], row['episode_part']))


    sub_datasets = {}
    for subject in other_subjects:
        friends_brain_data = get_subject_friends_data(subject, friends_scan_data_dir,test_set)
        localizer_data = get_subject_localizer_data(subject, task, localizer_dir, glm_single_all=True)
        stimuli_data = [model_class._get_stimuli_embeddings(friends_stimuli_dir, friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_brain_data]
        friends_episode_metadata = [(friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_brain_data]
        sub_dataset = SubjectTrialDataset(subject,[data['data'] for data in friends_brain_data], [data['data'] for data in localizer_data], stimuli_data, friends_episode_metadata, train_method='aggregate')
        sub_datasets[subject] = sub_dataset

    friends_held_out_test_data = get_subject_friends_data(held_out_subject, friends_scan_data_dir,test_set)
    friends_held_out_val_data = get_subject_friends_data(held_out_subject, friends_scan_data_dir,validation_set)
    localizer_data = get_subject_localizer_data(held_out_subject, task, localizer_dir, glm_single_all=True)
    stimuli_held_out_test_data = [model_class._get_stimuli_embeddings(friends_stimuli_dir, friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_held_out_test_data]
    stimuli_held_out_val_data = [model_class._get_stimuli_embeddings(friends_stimuli_dir, friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_held_out_val_data]
    friends_held_out_test_episode_metadata = [(friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_held_out_test_data]
    friends_held_out_val_episode_metadata = [(friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_held_out_val_data]

    held_out_test_dataset = SubjectTrialDataset(held_out_subject,[data['data'] for data in friends_held_out_test_data], [data['data'] for data in localizer_data], stimuli_held_out_test_data, friends_held_out_test_episode_metadata, train_method='aggregate')
    held_out_val_dataset = SubjectTrialDataset(held_out_subject,[data['data'] for data in friends_held_out_val_data], [data['data'] for data in localizer_data], stimuli_held_out_val_data, friends_held_out_val_episode_metadata, train_method='aggregate')

    test_in_dist_loader = DataLoader(held_out_val_dataset,shuffle=False, collate_fn=collate_fn)
    test_out_dist_loader = DataLoader(held_out_test_dataset, shuffle=False, collate_fn=collate_fn)

    best_ckpt = torch.load(f"{model_dir}/{held_out_subject}_best.pt", map_location=device)
    final_model = model_class(**best_ckpt['config']).to(device)
    final_model.load_state_dict(best_ckpt['model_state_dict'])
    final_model.eval()

    subject_maps = {}
    subject_glm_results = {}
    subject_reliability = {}
    subject_average_results = {}
    

    for subject in sub_datasets:
        subject_maps[subject] = {}
        subject_glm_results[subject] = {}
        subject_reliability[subject] = {}
        subject_average_results[subject] = {}
        trained_loader = DataLoader(sub_datasets[subject],shuffle=False,collate_fn=collate_fn)
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
                loss = np.corrcoef(outputs[i], targets[i])[0][1]
                ep_id = f"s{metadata[i][0]}e{metadata[i][1]}{metadata[i][2]}"
                subject_maps[subject][ep_id] = outputs[i]
                subject_glm_results[subject][ep_id] = loss
                
        #Now we have all the maps for a subject, we can compare them to each other
        ep_ids = list(subject_maps[subject].keys())
        for i in range(len(ep_ids)):
            subject_reliability[subject][ep_ids[i]] = {}
            for j in range(i+1,len(ep_ids)):
                subject_reliability[subject][ep_ids[i]][ep_ids[j]] = np.corrcoef(subject_maps[subject][ep_ids[i]], subject_maps[subject][ep_ids[j]])[0][1]
            subject_average_map = np.load(os.path.join(localizer_dir,subject,task,'average_mask.npy'))
            subject_average_results[subject][ep_ids[i]] = np.corrcoef(subject_maps[subject][ep_ids[i]], subject_average_map)[0][1]

    
    subject_maps[held_out_subject] = {'test': {}, 'validation': {}}
    subject_glm_results[held_out_subject] = {'test': {}, 'validation': {}}
    subject_reliability[held_out_subject] = {'test': {}, 'validation': {}}
    subject_average_results[held_out_subject] = {'test': {}, 'validation': {}}
    held_out_average_results = {'test': {'average_averaged': {}, 'glmsingle_averaged': {}}, 'validation': {'average_averaged': {}, 'glmsingle_averaged': {}}}
    for batch in test_in_dist_loader:

            fmri = batch['fmri'].to(device)
            stimulus = batch['stimulus'].to(device)
            targets = batch['localizer'].to(device)
            lengths = batch['lengths'].to(device)
            metadata = batch['metadata']
            outputs = final_model(fmri, stimulus, lengths)

            outputs = outputs.detach().cpu()
            targets = targets.detach().cpu()
            for i in range(targets.shape[0]):
                loss = np.corrcoef(outputs[i], targets[i])[0][1]
                ep_id = f"s{metadata[i][0]}e{metadata[i][1]}{metadata[i][2]}"
                subject_maps[held_out_subject]['validation'][ep_id] = outputs[i]
                subject_glm_results[held_out_subject]['validation'][ep_id] = loss
                
    #Now we have all the maps for a subject, we can compare them to each other
    ep_ids = list(subject_maps[held_out_subject]['validation'].keys())
    for i in range(len(ep_ids)):
        subject_reliability[held_out_subject]['validation'][ep_ids[i]] = {}
        for j in range(i+1,len(ep_ids)):
            subject_reliability[held_out_subject]['validation'][ep_ids[i]][ep_ids[j]] = np.corrcoef(subject_maps[held_out_subject]['validation'][ep_ids[i]], subject_maps[held_out_subject]['validation'][ep_ids[j]])[0][1]
        subject_average_map = np.load(os.path.join(localizer_dir,held_out_subject,task,'average_mask.npy'))
        subject_predicted_average_map = np.load(os.path.join(localizer_dir,held_out_subject,task,'average_averaged_mask.npy'))
        subject_predicted_glmsingle_average_map = np.load(os.path.join(localizer_dir,held_out_subject,task,'glm_single_averaged_mask.npy'))
        subject_average_results[held_out_subject]['validation'][ep_ids[i]] = np.corrcoef(subject_maps[held_out_subject]['validation'][ep_ids[i]], subject_average_map)[0][1]
        held_out_average_results['validation']['average_averaged'][ep_ids[i]] = np.corrcoef(subject_maps[held_out_subject]['validation'][ep_ids[i]], subject_predicted_average_map)[0][1]
        held_out_average_results['validation']['glmsingle_averaged'][ep_ids[i]] = np.corrcoef(subject_maps[held_out_subject]['validation'][ep_ids[i]], subject_predicted_glmsingle_average_map)[0][1]

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
                loss = np.corrcoef(outputs[i], targets[i])[0][1]
                ep_id = f"s{metadata[i][0]}e{metadata[i][1]}{metadata[i][2]}"
                subject_maps[held_out_subject]['test'][ep_id] = outputs[i]
                subject_glm_results[held_out_subject]['test'][ep_id] = loss
                
    #Now we have all the maps for a subject, we can compare them to each other
    ep_ids = list(subject_maps[held_out_subject]['test'].keys())
    for i in range(len(ep_ids)):
        subject_reliability[held_out_subject]['test'][ep_ids[i]] = {}
        for j in range(i+1,len(ep_ids)):
            subject_reliability[held_out_subject]['test'][ep_ids[i]][ep_ids[j]] = np.corrcoef(subject_maps[held_out_subject]['test'][ep_ids[i]], subject_maps[held_out_subject]['test'][ep_ids[j]])[0][1]
        subject_average_map = np.load(os.path.join(localizer_dir,held_out_subject,task,'average_mask.npy'))
        subject_predicted_average_map = np.load(os.path.join(localizer_dir,held_out_subject,task,'average_averaged_mask.npy'))
        subject_predicted_glmsingle_average_map = np.load(os.path.join(localizer_dir,held_out_subject,task,'glm_single_averaged_mask.npy'))
        subject_average_results[held_out_subject]['test'][ep_ids[i]] = np.corrcoef(subject_maps[held_out_subject]['test'][ep_ids[i]], subject_average_map)[0][1]
        held_out_average_results['test']['average_averaged'][ep_ids[i]] = np.corrcoef(subject_maps[held_out_subject]['test'][ep_ids[i]], subject_predicted_average_map)[0][1]
        held_out_average_results['test']['glmsingle_averaged'][ep_ids[i]] = np.corrcoef(subject_maps[held_out_subject]['test'][ep_ids[i]], subject_predicted_glmsingle_average_map)[0][1]


    inter_subject_consistency = {'same_episode': {}, 'diff_episode': {}}
    for i in range(10):
        ep_id = list(subject_maps[held_out_subject]['test'].keys())[i]
        #ep_id_2 = list(subject_maps[held_out_subject]['test'].keys())[i+1]
        subjects = list(subject_maps.keys())
        inter_subject_consistency['same_episode'][ep_id] = {}
        #print('\t\t',ep_id,'vs',ep_id_2)
        for j in range(len(subjects)):
            subject_j = subjects[j]
            inter_subject_consistency['same_episode'][ep_id][subject_j] = {}
            for k in range(j+1,len(subjects)):
                #subject_j = subjects[j]
                subject_k = subjects[k]
                
                if subject_j == held_out_subject:
                    subject_j_map = subject_maps[subject_j]['test'][ep_id]
                else:
                    subject_j_map = subject_maps[subject_j][ep_id]

                if subject_k == held_out_subject:
                    subject_k_map = subject_maps[subject_k]['test'][ep_id]
                else:
                    subject_k_map = subject_maps[subject_k][ep_id]
                inter_subject_consistency['same_episode'][ep_id][subject_j][subject_k] = np.corrcoef(subject_j_map,subject_k_map)[0][1]
                #print(np.corrcoef(subject_j_map,subject_k_map))
    
    for i in range(10):
        ep_id = list(subject_maps[held_out_subject]['test'].keys())[i]
        ep_id_2 = list(subject_maps[held_out_subject]['test'].keys())[-1*i]
        subjects = list(subject_maps.keys())
        inter_subject_consistency['diff_episode'][f"{ep_id}_vs_{ep_id_2}"] = {}
        #print('\t\t',ep_id,'vs',ep_id_2)
        for j in range(len(subjects)):
            subject_j = subjects[j]
            inter_subject_consistency['diff_episode'][f"{ep_id}_vs_{ep_id_2}"][subject_j] = {}
            for k in range(j+1,len(subjects)):
                #subject_j = subjects[j]
                subject_k = subjects[k]
                
                if subject_j == held_out_subject:
                    subject_j_map = subject_maps[subject_j]['test'][ep_id]
                else:
                    subject_j_map = subject_maps[subject_j][ep_id]

                if subject_k == held_out_subject:
                    subject_k_map = subject_maps[subject_k]['test'][ep_id_2]
                else:
                    subject_k_map = subject_maps[subject_k][ep_id_2]
                inter_subject_consistency['diff_episode'][f"{ep_id}_vs_{ep_id_2}"][subject_j][subject_k] = np.corrcoef(subject_j_map,subject_k_map)[0][1]
               


    examples_dir = f"{model_dir}/{held_out_subject}_brain_maps/"
    for i in range(5):
        ep_id = list(subject_maps[held_out_subject]['test'].keys())[i]
        for subject in subject_maps:
            subject_example_dir = os.path.join(examples_dir, subject)
            os.makedirs(subject_example_dir, exist_ok=True)
            if subject == held_out_subject:
                np.save(f"{subject_example_dir}/{ep_id}.npy", subject_maps[subject]['test'][ep_id])
            else:
                np.save(f"{subject_example_dir}/{ep_id}.npy", subject_maps[subject][ep_id])


    return {
            'glm_correlations': subject_glm_results,
            'intra_subject_reliability_correlations': subject_reliability,
            'inter_subject_average_correlations': subject_average_results,
            'held_out_subject_predictability_correlations': held_out_average_results,
            'inter_subject_episode_map_consistency': inter_subject_consistency
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
    args = parser.parse_args()

    localizer_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/mask/{args.parcellation}"
    model_dir = f"{checkpoint_base_dir}/{args.parcellation}/{args.name}/"
    

    model_types = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir,f))]
    for model_type in model_types:
        tasks = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir,f))]
        subjects = [f'sub-0{i}' for i in range(1,7)]
        results = {}
        for task in tasks:
            print('Running task', task)
            results[task] = {}
            task_dir = os.path.join(model_dir,task)
            for held_out_subject in subjects:
                print('\t on', held_out_subject)
                results[task][held_out_subject] = evaluate_with_held_out_subject(held_out_subject, task, task_dir, localizer_dir, friends_scan_data_dir, friends_stimuli_dir)

    with open(os.path.join(model_dir,f"{args.name}_{model_class.MODEL_NAME}_results.json"),"w") as f:
        json.dump(results, f)


if __name__=="__main__":
    main()
