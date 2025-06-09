from argparse import ArgumentParser
from dotenv import load_dotenv
from datetime import datetime
from subject_dataset import SubjectTrialDataset
from model_to_run import model_class

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
import scipy.stats as st
from sklearn import metrics

import optuna
from torch.cuda.amp import autocast, GradScaler

# At the top of the file, after imports
def setup_logging(base_dir, task, sub_id=None,parcellation='tian'):
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


def get_subject_friends_data(sub_id, friends_scan_path):
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


def make_dataloaders(train_ids, val_ids, root_dir, batch_size=32):
    train_datasets = load_subject_datasets(train_ids, root_dir)
    val_datasets = load_subject_datasets(val_ids, root_dir)

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ---- Early Stopping ----
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True


# ---- Training with Early Stopping ----
def train_with_early_stopping(model, train_loader, val_loader, optimizer, device, loss_fn=nn.MSELoss(), num_epochs=20, patience=5, delta=0,checkpoint_dir=None, checkpoint_prefix="model"):
    model = model.to(device)
    early_stopper = EarlyStopping(patience=patience, min_delta=delta)
    best_val_loss = float('inf')

    scheduler = None

    scaler = GradScaler(enabled=False)  # Disable gradient scaling for BF16
    if val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    for epoch in range(num_epochs):
        logging.info(f"    Epoch {epoch+1}/{num_epochs}")
        model.train()
        if hasattr(train_loader.dataset, 'shuffle_localizers'):
            train_loader.dataset.shuffle_localizers()

        for batch in tqdm(train_loader):
            fmri = batch['fmri'].to(device)
            stimulus = batch['stimulus'].to(device)
            #stimulus = None
            targets = batch['localizer'].to(device)
            #print("fmri:", fmri.min().item(), fmri.max().item())
            #print("stimulus:", stimulus.min().item(), stimulus.max().item())
            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16):
                outputs = model(fmri, stimulus, batch['lengths'].to(device))
                #print("OUTPUT:", outputs.min().item(), outputs.max().item())
                #print("TARGET:", targets.min().item(), targets.max().item())
                loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    fmri = batch['fmri'].to(device)
                    stimulus = batch['stimulus'].to(device)
                    #stimulus = None
                    targets = batch['localizer'].to(device)
                    with autocast(dtype=torch.bfloat16):
                        outputs = model(fmri, stimulus, batch['lengths'].to(device))
                        val_loss = loss_fn(outputs, targets)
                    val_loss_total += val_loss.item()
            avg_val_loss = val_loss_total / len(val_loader)
            early_stopper(avg_val_loss)
            if scheduler:
                scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if checkpoint_dir is not None:
                    checkpoint_path = f"{checkpoint_dir}/{checkpoint_prefix}_best.pt"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': avg_val_loss,
                        'config': model.config,
                    }, checkpoint_path)

            if early_stopper.should_stop:
                logging.info(f"Early stopping at epoch {epoch+1} with val loss {avg_val_loss:.4f}")
                break
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU peak memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved : {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    return best_val_loss


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


# ---- Full CV with Random Hyperparameter Search ----
def run_cross_validation_with_tuning(
        subject_datasets, 
        model_class, 
        val_episodes, 
        test_episodes, 
        optimizer_class=optim.AdamW, 
        checkpoint_dir='checkpoints',
        held_out_sub=None, 
        batch_size=4, 
        device='cuda', 
        num_trials=10, 
        num_epochs=20, 
        patience=5, 
        delta=0,
        parcellation_size=400
        ):
    subject_ids = list(subject_datasets.keys())

    if held_out_sub:
        test_ids = [held_out_sub]
    else:
        test_ids = subject_ids


        # If episode holdout is requested, split datasets
    
    def split_trials_by_episode(fmri_trials, stim_labels, localizer_maps, friends_episode_metadata, val_episodes,test_episodes):
            train_fmri, train_labels, train_metadata, val_fmri, val_labels, val_metadata, test_fmri, test_labels, test_metadata = [], [], [], [], [], [],[],[],[]
            for x, stim, metadata in zip(fmri_trials, stim_labels, friends_episode_metadata):
                #print(metadata)
                #raise ValueError('e')
                episode = get_episode_from_metadata(metadata)
                if episode in val_episodes:
                    val_fmri.append(x)
                    val_labels.append(stim)
                    val_metadata.append(metadata)
                elif episode in test_episodes:
                    test_fmri.append(x)
                    test_labels.append(stim)
                    test_metadata.append(metadata)
                else:
                    train_fmri.append(x)
                    train_labels.append(stim)
                    train_metadata.append(metadata)
            return (train_fmri, train_labels, train_metadata), (val_fmri, val_labels, val_metadata), (test_fmri, test_labels, test_metadata), localizer_maps

    train_subjects = {}
    val_subjects = {}
    test_subjects = {}
    for sid, dataset in subject_datasets.items():
        (train_fmri, train_labels, train_metadata), (val_fmri, val_labels, val_metadata), (test_fmri, test_labels, test_metadata), localizers = split_trials_by_episode(
            dataset.fmri_list, dataset.stimuli_list, dataset.localizers, dataset.friends_episode_metadata, val_episodes, test_episodes)
        #print(train_metadata)
        #print(val_metadata)
        #raise ValueError('e')
        train_subjects[sid] = SubjectTrialDataset(sid,train_fmri, localizers,train_labels, train_metadata, dataset.train_method)
        val_subjects[sid] = SubjectTrialDataset(sid,val_fmri, localizers,val_labels, val_metadata, dataset.train_method)
        test_subjects[sid] = SubjectTrialDataset(sid,test_fmri, localizers,test_labels, test_metadata, dataset.train_method)
        #print(train_subjects[sid])
        #print(val_subjects[sid])

    for test_id in test_ids:
        logging.info(f"\nOuter Fold: Test on Subject {test_id}")
        inner_ids = [sid for sid in subject_ids if sid != test_id]

        test_data_in_distribution = ConcatDataset([train_subjects[test_id],val_subjects[test_id]])
        test_data_out_distribution = test_subjects[test_id]

        best_config = None
        best_val_loss = float('inf')


        def objective(trial):
#            config = {
#                'aggregation_method': trial.suggest_categorical('aggregation_method', ['mean', 'lstm', 'attention']),
#                'lr': 10 ** trial.suggest_float('lr_exp', -5, -3),
#                'embedding_dim': trial.suggest_categorical('embedding_dim', [16, 32, 64]),
#                'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
#                'transformer_layers': trial.suggest_categorical('transformer_layers', [2, 4, 6]),
#                'weight_decay': trial.suggest_categorical('weight_decay', [0.0, 1e-5, 1e-4])
#            }
            config = model_class._sample_optuna_hyperparams(trial)
            total_val_loss = 0.0

            for val_id in inner_ids:
                train_ids = [sid for sid in inner_ids if sid != val_id]
                train_dataset = ConcatDataset([train_subjects[sid] for sid in train_ids])
                val_dataset = train_subjects[val_id]

                subject_lengths = {sid: len(train_subjects[sid]) for sid in train_ids}
                sample_weights = []
                for sid in train_ids:
                    weight = 1.0 / subject_lengths[sid]
                    sample_weights.extend([weight] * subject_lengths[sid])
                sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

                model = model_class(surface_dim=parcellation_size,**config)
                optimizer = optimizer_class(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
                val_loss = train_with_early_stopping(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, patience=patience,delta=delta)
                total_val_loss += val_loss

            avg_val_loss = total_val_loss / len(inner_ids)
            return avg_val_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=num_trials)

        #best_config = {
        #    'aggregation_method': study.best_trial.params['aggregation_method'],
        #    'lr': study.best_trial.params['lr_exp'],
        #    'embedding_dim': study.best_trial.params['embedding_dim'],
        #    'num_heads': study.best_trial.params['num_heads'],
        #    'transformer_layers': study.best_trial.params['transformer_layers'],
        #    'weight_decay': study.best_trial.params['weight_decay'],
        #    'input_projection_size': study.best_trial.params['input_projection_size'],
        #    'dropout_rate': study.best_trial.params['dropout_rate'],
        #    'fc_output_size': study.best_trial.params['input_projection_size'],
        #}
        best_config = model_class._found_hyperparams(study)

        logging.info(f"Best config for Subject {test_id}: {best_config}")
        
        final_train_dataset = ConcatDataset([train_subjects[sid] for sid in inner_ids])

        final_subject_lengths = {sid: len(train_subjects[sid]) for sid in inner_ids}
        final_sample_weights = []
        for sid in inner_ids:
            w = 1.0 / final_subject_lengths[sid]
            final_sample_weights.extend([w] * final_subject_lengths[sid])
        final_sampler = WeightedRandomSampler(final_sample_weights, num_samples=len(final_sample_weights), replacement=True)
        final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, sampler=final_sampler, collate_fn=collate_fn)
#        final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        final_model = model_class(surface_dim=parcellation_size,**best_config)
        final_optimizer = optimizer_class(final_model.parameters(), lr=best_config['lr'], weight_decay=best_config.get('weight_decay', 0.0))

        logging.info("Retraining on full training set with fixed number of epochs (no val set)...")
        val_dataset = ConcatDataset([val_subjects[sid] for sid in inner_ids])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        train_with_early_stopping(
                final_model, 
                final_train_loader, 
                val_loader, 
                final_optimizer, 
                device, 
                num_epochs=num_epochs, 
                patience=patience,
                delta=delta,
                checkpoint_dir=checkpoint_dir,
                checkpoint_prefix=f"{test_id}"
        )

        
        #os.makedirs(checkpoint_dir, exist_ok=True)
        #torch.save({
        #                'model_state_dict': model.state_dict(),
        #                'config': model.config,
        #                'hyperparameters': best_config
        #}, f"{checkpoint_dir}/{test_id}_final_model.pt")


        ### --- Evaluate on held-out test subject ---
        test_in_dist_loader = DataLoader(test_data_in_distribution, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_out_dist_loader = DataLoader(test_data_out_distribution, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Reload best checkpoint
        best_ckpt = torch.load(f"{checkpoint_dir}/{test_id}_best.pt", map_location=device)
        final_model.load_state_dict(best_ckpt['model_state_dict'])
        final_model.eval()

        total_test_loss = 0.0
        loss_fn = nn.MSELoss()

        # save individual results?
        # run comparisons between methods of trained individual: average, tmins, glmsingle
        # run comparisons between methods of held-out-individuals: average_average, glmsingle
        
        subject_results = {}
        with torch.no_grad():
            subject_results[test_id] = []
            for batch in test_out_dist_loader:
                
                fmri = batch['fmri'].to(device)
                stimulus = batch['stimulus'].to(device)
                targets = batch['localizer'].to(device)
                lengths = batch['lengths'].to(device)
                metadatas = batch['metadata']
                outputs = final_model(fmri, stimulus, lengths)
                
                outputs = outputs.detach().cpu()
                targets = targets.detach().cpu()
                for i in range(targets.shape[0]):
                    loss = metrics.r2_score(targets[i],outputs[i])#np.corrcoef(outputs[i], targets[i])
                    print(metadatas[i],loss)
                    #print(loss.shape)
                    subject_results[test_id].append({
                        "set": "test",
                        "season": metadatas[i][0],
                        "episode": metadatas[i][1],
                        "episode_part": metadatas[i][2],
                        "correlation": loss
                    })
                    total_test_loss += loss

        avg_test_loss = total_test_loss / len(test_data_out_distribution)
        logging.info(f"Test metric for {test_id}: {avg_test_loss:.4f}")

        total_test_loss = 0
        with torch.no_grad():
            for batch in test_in_dist_loader:
                
                fmri = batch['fmri'].to(device)
                stimulus = batch['stimulus'].to(device)
                targets = batch['localizer'].to(device)
                lengths = batch['lengths'].to(device)
                metadatas = batch['metadata']
                outputs = final_model(fmri, stimulus, lengths)
                outputs = outputs.detach().cpu()
                targets = targets.detach().cpu()
                for i in range(targets.shape[0]):
                    loss = metrics.r2_score(targets[i],outputs[i])#np.corrcoef(outputs[i], targets[i])
                    print(metadatas[i],loss)
                    subject_results[test_id].append({
                        "set": "validation",
                        "season": metadatas[i][0],
                        "episode": metadatas[i][1],
                        "episode_part": metadatas[i][2],
                        "correlation": loss
                    })
                    total_test_loss += loss

        avg_test_loss = total_test_loss / len(test_data_in_distribution)
        logging.info(f"Test metric for {test_id}: {avg_test_loss:.4f}")

        total_test_loss = 0
        for sid in inner_ids:
            subject_results[sid] = []
            #val_dataset = ConcatDataset([val_subjects[sid] for sid in inner_ids])
            test_loader = DataLoader(test_subjects[sid], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            with torch.no_grad():
                for batch in test_loader:
                
                    fmri = batch['fmri'].to(device)
                    stimulus = batch['stimulus'].to(device)
                    targets = batch['localizer'].to(device)
                    lengths = batch['lengths'].to(device)
                    metadatas = batch['metadata']
                    outputs = final_model(fmri, stimulus, lengths)
                    outputs = outputs.detach().cpu()
                    targets = targets.detach().cpu()
                    for i in range(targets.shape[0]):
                        loss = metrics.r2_score(targets[i],outputs[i])#np.corrcoef(outputs[i], targets[i])
                        subject_results[sid].append({
                            "set": "validation",
                            "season": metadatas[i][0],
                            "episode": metadatas[i][1],
                            "episode_part": metadatas[i][2],
                            "correlation": loss
                        })
                        print(sid,metadatas[i],loss)
                        total_test_loss += loss

        avg_test_loss = total_test_loss / sum([len(test_subjects[sid]) for sid in inner_ids])
        logging.info(f"Test metric for {test_id}: {avg_test_loss:.4f}")
        
        
        dataset_info = []
        for batch in test_in_dist_loader:
            metadatas = batch['metadata']
            for i in range(len(metadatas)):
                dataset_info.append({
                    'season': metadatas[i][0],
                    'episode': metadatas[i][1],
                    'episode_part': metadatas[i][2],
                    'set': 'validation'
                    })
        for batch in test_out_dist_loader:
            metadatas = batch['metadata']
            for i in range(len(metadatas)):
                dataset_info.append({
                    'season': metadatas[i][0],
                    'episode': metadatas[i][1],
                    'episode_part': metadatas[i][2],
                    'set': 'test'
                    })


        with open(f"{checkpoint_dir}/{test_id}_validation_and_test_set.csv","w") as f:
            fieldnames = ['season','episode','episode_part','set']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for data in dataset_info:
                writer.writerow(data)
            #json.dump(subject_results, f)

        logging.info(f"Finished fold for Subject {test_id}")


def main():
    load_dotenv()
    friends_scan_data_dir = os.getenv("FRIENDS_SCAN_DATA_DIR")
    friends_stimuli_dir = os.getenv("FRIENDS_STIMULI_DIR")
    base_dir = os.getenv("FRIENDS_BASE_DIR")
    scratch_dir = os.getenv("FRIENDS_SCRATCH_DIR")
    checkpoint_base_dir = os.getenv("FRIENDS_CHECKPOINT_DIR")
    #torch.cuda.set_per_process_memory_fraction(.48, device=0)

    print('scan data dir', friends_scan_data_dir)
    print('stimuli dir', friends_stimuli_dir)
    print('base dir', base_dir)
    print('scratch dir', scratch_dir)
    print('checkpoint base dir', checkpoint_base_dir)

    parser = ArgumentParser(description="train model on fmri data")
    parser.add_argument("sub_id", help="Subject ID", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--task", help="Task", type=str, default="wm")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0)
    parser.add_argument("--parcellation", type=str, default="tian")

    args = parser.parse_args()
    if args.parcellation not in ['tian','4s456','4s656','4s1056']:
        raise ValueError(f"parcellation {args.parcellation} is not supported")
    
    parcellation_size_map = {
            'tian': 400,
            '4s456': 456,
            '4s656': 656,
            '4s1056': 1056
            }

    localizer_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/mask/{args.parcellation}"
    friends_scan_data_dir = os.path.join(friends_scan_data_dir,args.parcellation)

    torch.set_float32_matmul_precision('high')  # Optimizes matmul on H100
    setup_logging(base_dir, args.task, args.sub_id,args.parcellation)

    subject_ids = [f'sub-0{i}' for i in range(1,7)]
    subject_datasets = {}
    for subject_id in subject_ids:
        friends_brain_data = get_subject_friends_data(subject_id, friends_scan_data_dir)
        #localizer_data = get_subject_localizer_data(subject_id, args.task, localizer_dir)
        stimuli_data = [model_class._get_stimuli_embeddings(friends_stimuli_dir, friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_brain_data]
        friends_episode_metadata = [(friends_episode_data['season'], friends_episode_data['episode_num'], friends_episode_data['episode_part']) for friends_episode_data in friends_brain_data]
        if subject_id == args.sub_id:
            localizer_data = get_subject_localizer_data(subject_id, args.task, localizer_dir, args.parcellation,binary=False)
            sub_dataset = SubjectTrialDataset(subject_id,[data['data'] for data in friends_brain_data], [data['data'] for data in localizer_data], stimuli_data, friends_episode_metadata, train_method='aggregate')
        else:
            localizer_data = get_subject_localizer_data(subject_id, args.task, localizer_dir, args.parcellation,binary=False)
            sub_dataset = SubjectTrialDataset(subject_id,[data['data'] for data in friends_brain_data], [data['data'] for data in localizer_data], stimuli_data, friends_episode_metadata, train_method='aggregate')
        subject_datasets[subject_id] = sub_dataset

    checkpoint_dir = os.path.join(checkpoint_base_dir, args.parcellation, args.name, model_class.MODEL_NAME, args.task)

    held_out_episodes = np.random.permutation(np.arange(25,97))

    run_cross_validation_with_tuning(
            subject_datasets,
            model_class,
            val_episodes=held_out_episodes[:10],
            test_episodes=held_out_episodes[10:20],
            checkpoint_dir=checkpoint_dir,
            held_out_sub=args.sub_id,
            batch_size=args.batch_size,
            num_trials=args.num_trials,
            num_epochs=args.num_epochs,
            patience=args.patience,
            delta=args.delta,
            parcellation_size=parcellation_size_map[args.parcellation]
    )

if __name__=="__main__":
    main()
