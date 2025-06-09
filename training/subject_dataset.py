import torch
import numpy as np
import json
import math
from dotenv import load_dotenv
from argparse import ArgumentParser

import nibabel as nib

class SubjectTrialDataset(torch.utils.data.Dataset):
    def __init__(self, subject_id,fmri_list, localizers, stimuli_list=None, friends_episode_metadata=None, train_method='sample'):
        """
        fmri_list: list of tensors of shape (T_i, 91248 or # parcels)
        stim_list: list of stimulus labels (ints)
        localizer: tensor of shape (91248,) or (# parcels,) — same for all trials
        """
        self.fmri_list = fmri_list
        self.subject_id = subject_id
        self.friends_episode_metadata = friends_episode_metadata
        
#        print(friends_episode_metadata)
#        print(friends_episode_metadata[0])
        if stimuli_list:
            self.stimuli_list = stimuli_list
        else:
            self.stimuli_list = [None for _ in range(len(self.fmri_list))]
        self.train_method = train_method
        assert train_method in ['exhaustive', 'sample', 'aggregate'], f"Training method {train_method} not recognized"
        if train_method == 'aggregate':
            self.localizers = localizers if len(localizers)==1 else [np.mean(localizers, axis=0)]
        else:
            self.localizers = localizers

        if train_method == 'sample':
            self._new_localizer_idxs()

        atlas_dlabel_path='/orcd/data/satra/001/users/jsmentch/atlases/atlas-Glasser/atlas-Glasser_space-fsLR_den-91k_dseg.dlabel.nii'
        img = nib.load(atlas_dlabel_path)
        atlas_data=img.get_fdata()
        self.atlas_data=atlas_data[0,:]

        self.parcel_ids = set(self.atlas_data.tolist())


    def _new_localizer_idxs(self):
        self.localizer_idxs = np.random.permutation(len(self.localizers))
        self.current_localizer_idx = 0

    def __len__(self):
        if self.train_method != 'exhaustive':
            return len(self.fmri_list)
        else:
            return len(self.fmri_list)*len(self.localizers)


    def _parcellate(self, data):
        new_data = np.zeros((data.shape[0],len(self.parcel_ids)-1))
        for parcel_id in self.parcel_ids:
            if parcel_id == 0:
                continue
            voxels_in_parcel = np.where(self.atlas_data==parcel_id)[0]
            new_data[:,int(parcel_id)-1] = np.average(data[:,voxels_in_parcel],axis=1)
        return new_data

    def __getitem__(self, idx):
        if self.train_method == 'exhaustive':
            fmri_idx = idx//len(self.localizers)
            localizer_idx = idx%len(self.localizers)
        elif self.train_method == 'sample':
            fmri_idx = idx
            localizer_idx = self.localizer_idxs[self.current_localizer_idx]
            if self.current_localizer_idx == len(self.localizer_idxs)-1:
                self._new_localizer_idxs()
        elif self.train_method == 'aggregate':
            fmri_idx = idx
            localizer_idx = 0
        else:
            raise ValueError(f"Training method {self.training_method} not recognized")

        #fmri_data = nib.load(self.fmri_list[fmri_idx]).get_fdata()
        #atlas_dlabel='/orcd/data/satra/001/users/jsmentch/atlases/atlas-Glasser/atlas-Glasser_space-fsLR_den-91k_dseg.dlabel.nii'
        fmri_data = self.fmri_list[fmri_idx]#self._parcellate(fmri_data)
        #e = f"{fmri_idx}, {len(self.fmri_list)}, {len(self.localizers)},{len(self.stimuli_list)},{len(self.friends_episode_metadata)}, {self.subject_id}"
        #raise ValueError(e)
    # 240, 252, 252,14,252, sub-05
        #print(self.friends_episode_metadata[fmri_idx])
        #print(len(self.friends_episode_metadata))
        #raise ValueError('stop')

        return {
            "fmri": torch.tensor(fmri_data,dtype=torch.float32),  # Variable length sequence
            "localizer": torch.tensor(self.localizers[localizer_idx],dtype=torch.float32),  # Same target for all trials
            "stimulus": torch.tensor(self.stimuli_list[fmri_idx]),
            "metadata": self.friends_episode_metadata[fmri_idx]
        }
"""
def main():
    load_dotenv()
    friends_scan_data_dir = os.getenv("FRIENDS_SCAN_DATA_DIR")
    base_dir = os.getenv("FRIENDS_BASE_DIR")
    scratch_dir = os.getenv("FRIENDS_SCRATCH_DIR")

    parser = ArgumentParser(description="get baseline statistics using GLMsingle results")
    parser.add_argument("sub_id", help="Subject ID", type=str)
    parser.add_argument("--task", help="Task", type=str, default="wm")
    args = parser.parse_args()


if __name__=="__main__":
    main()
"""
