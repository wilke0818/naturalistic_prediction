import os
from dotenv import load_dotenv
from scipy import stats
import numpy as np
from argparse import ArgumentParser
from statsmodels.stats.multitest import fdrcorrection
import logging
from datetime import datetime
import math
import warnings

import nibabel as nib
import hcp_utils as hcp
def voxel_parcellation(betas, signal_voxels, parcellation='glasser'):

    if parcellation=='glasser':
        atlas_dlabel='/orcd/data/satra/001/users/jsmentch/atlases/atlas-Glasser/atlas-Glasser_space-fsLR_den-91k_dseg.dlabel.nii'
        img = nib.load(atlas_dlabel)
        atlas_data=img.get_fdata()
        atlas_data=atlas_data[0,:]
        atlas_parcel_remap = {}
    elif parcellation=='4s456':
        atlas_dlabel = '/orcd/data/satra/001/users/jsmentch/atlases/atlas-4S456Parcels/atlas-4S456Parcels_space-fsLR_den-91k_dseg.dlabel.nii'
        img = nib.load(atlas_dlabel)
        atlas_data=img.get_fdata()
        atlas_data=atlas_data[0,:]
        atlas_parcel_remap = {}#{533:532, 903:905}
    elif parcellation=='4s656':
        #atlas_dlabel='/home/wilke18/Schaefer2018_900Parcels_Kong2022_17Networks_order.dscalar.nii'
        #atlas_dlabel = '/home/wilke18/Schaefer2018_400Parcels_17Networks_order_Tian_Subcortex_S4.dlabel.nii'
        #atlas_dlabel = '/orcd/data/satra/001/users/jsmentch/atlases/atlas-4S1056Parcels/atlas-4S1056Parcels_space-fsLR_den-91k_dseg.dlabel.nii'
        atlas_dlabel = '/orcd/data/satra/001/users/jsmentch/atlases/atlas-4S656Parcels/atlas-4S656Parcels_space-fsLR_den-91k_dseg.dlabel.nii'
        img = nib.load(atlas_dlabel)
        atlas_data=img.get_fdata()
        atlas_data=atlas_data[0,:]
        atlas_parcel_remap = {}#{533:532, 903:905}
    elif parcellation=='4s1056':
        atlas_dlabel = '/orcd/data/satra/001/users/jsmentch/atlases/atlas-4S1056Parcels/atlas-4S1056Parcels_space-fsLR_den-91k_dseg.dlabel.nii'
        img = nib.load(atlas_dlabel)
        atlas_data=img.get_fdata()
        atlas_data=atlas_data[0,:]
        atlas_parcel_remap = {533:532, 903:905}
    elif parcellation=='tian':
        atlas_dlabel = '/home/wilke18/Schaefer2018_400Parcels_17Networks_order_Tian_Subcortex_S4.dlabel.nii'
        img = nib.load(atlas_dlabel)
        atlas_data=img.get_fdata()
        atlas_data=atlas_data[0,:]
        atlas_parcel_remap = {}#{533:532, 903:905}
    else:
        raise ValueError(f"Unrecognized parcellation {parcellation}")
    parcels = np.zeros((int(max(atlas_data)),betas.shape[1]))
    logging.info(f"Creating parcel betas of shape {parcels.shape}")
    for parcel_id in set(atlas_data):
        if parcel_id == 0 or parcel_id in atlas_parcel_remap:
            continue
        voxels_in_parcel = np.where(atlas_data == parcel_id)[0]
        parcel = []
        for voxel in voxels_in_parcel:
            if voxel in signal_voxels:
                parcel.append(betas[voxel,:])

        parcels[int(parcel_id)-1] = np.sum(np.array(parcel),axis=0)/len(voxels_in_parcel)
    
    for parcel in atlas_parcel_remap:
        new_parcel = atlas_parcel_remap[parcel]
        parcels[parcel-1] = parcels[new_parcel-1]
    return parcels

# At the top of the file, after imports
def setup_logging(base_dir, sub_id, task):
    """Set up logging to both file and console."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(base_dir, 'logs', 'GLMsingle_localizer')
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{sub_id}_{task}_{timestamp}.log')

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

def get_stim_results(results, unique_stims, stim_type_order):
    """Extract beta values for each stimulus type."""
    stim_arrays = {}
    for stim in unique_stims:
        stim_indices = np.where(np.array(stim_type_order) == stim)[0]
        stim_arrays[stim] = results[:, stim_indices]
        logging.info(f"Extracted {len(stim_indices)} trials for stimulus type: {stim}")
    return stim_arrays

def main(sub_id, task, glm_results, designinfo, stim_files, output_dir, run=None,parcellation=None):
    logging.info(f"Processing subject {sub_id}, task {task}, run {run}, parcellation={parcellation}")
    
    # Load and validate data
    typed = glm_results['typed'].item()
    data = np.squeeze(typed["betasmd"])  # Shape: (91282, trials)
    logging.info(f"Loaded beta data with shape: {data.shape}")
    
    # Create stimulus mapping
    stim_type = sorted(list(set(stim_files)))
    stim_mapping = {i: stim_type[i] for i in range(len(stim_type))}
    stim_type_order = [stim_mapping[sid] for sid in designinfo['stimorder']]
    unique_stims = np.unique(stim_type_order)
    logging.info(f"Found stimulus types: {unique_stims}")

    # Extract stimulus-specific data
    stim_arrays = get_stim_results(data, unique_stims, stim_type_order)
    
    # Create signal mask
    noisepool = np.squeeze(typed["noisepool"])
    signal_mask = ~noisepool.astype(bool)
    n_signal_vertices = np.sum(signal_mask)
    logging.info(f"Signal vertices: {n_signal_vertices} out of {data.shape[0]}")

    # Extract signal data for each stimulus
    signal_arrays = {}
    for stim, beta_array in stim_arrays.items():
        signal_arrays[stim] = beta_array[signal_mask, :]
        logging.info(f"Signal array shape for {stim}: {signal_arrays[stim].shape}")
   
    if parcellation:
        # Prepare data for statistical comparison
        tool = np.concatenate((stim_arrays[unique_stims[3]],stim_arrays[unique_stims[7]]),axis=1)
        others = np.concatenate((stim_arrays[unique_stims[0]],stim_arrays[unique_stims[2]],stim_arrays[unique_stims[1]],stim_arrays[unique_stims[4]],stim_arrays[unique_stims[6]],stim_arrays[unique_stims[5]]),axis=1)

    
        # Only compute statistics for signal vertices
        signal_indices = np.where(signal_mask)[0]
        tool_parcels = voxel_parcellation(tool, signal_indices,parcellation)
        others_parcels = voxel_parcellation(others,signal_indices,parcellation)
        
        t_stats_full = np.zeros(tool_parcels.shape[0])  # Initialize with full brain size
        p_values_full = np.ones(tool_parcels.shape[0])-.00000001

        average_betas_stimuli = np.zeros((tool_parcels.shape[0],tool.shape[1]))
        average_betas_contrast = np.zeros((others_parcels.shape[0],others.shape[1]))
        logging.info(f"parcel shape: {tool_parcels.shape} and {others_parcels.shape}")

        count_warnings = 0
        for signal_idx in range(tool_parcels.shape[0]):
            t_stat, p_val = stats.ttest_ind(tool_parcels[signal_idx, :], others_parcels[signal_idx, :],alternative='greater')
            if math.isnan(t_stat) or math.isnan(p_val):
                t_stats_full[signal_idx] = 0
                p_values_full[signal_idx] = 1-.00000001
                count_warnings += 1
                warnings.warn(f"t_stat is {t_stat} and p_val is {p_val} at {signal_idx}, {tool_parcels[signal_idx, :]} > {others_parcels[signal_idx, :]}",RuntimeWarning)
            else:
                t_stats_full[signal_idx] = t_stat
                p_values_full[signal_idx] = p_val
            average_betas_stimuli[signal_idx] = tool_parcels[signal_idx, :]
            average_betas_contrast[signal_idx] = others_parcels[signal_idx, :]
    
        binary_mask = np.zeros(tool_parcels.shape[0])
        rejected, corrected_p = fdrcorrection(p_values_full, alpha=0.05)
        binary_mask[np.where((rejected) & (t_stats_full > 0))[0]] = 1

        average_betas_tool = np.average(average_betas_stimuli,axis=1)
        average_betas_others = np.average(average_betas_contrast,axis=1)

        warnings.warn(f"Saw {count_warnings} warnings", RuntimeWarning)
        stats_dict = {
            'total_vertices': data.shape[0],
            'signal_mask': signal_mask,
            'signal_vertices': n_signal_vertices,
            't_stats': t_stats_full,
            'p_values': p_values_full,
            'binary_mask': binary_mask,
            'p_values_corrected': corrected_p,
            'betas_stimuli': average_betas_tool,
            'betas_contrast': average_betas_others
        }
        parcel = parcellation if parcellation else 'unparcellated'
        stats_file = os.path.join(output_dir, f"{sub_id}_{task}_stats_{parcel}.npz")
        np.savez(stats_file, **stats_dict)
        logging.info(f"Saved statistics to: {stats_file}")
    else:
        # Prepare data for statistical comparison
        tool = np.concatenate((signal_arrays[unique_stims[3]],signal_arrays[unique_stims[7]]),axis=1)
        others = np.concatenate((signal_arrays[unique_stims[0]],signal_arrays[unique_stims[2]],signal_arrays[unique_stims[1]],signal_arrays[unique_stims[4]],signal_arrays[unique_stims[6]],signal_arrays[unique_stims[5]]),axis=1)

        # Statistical testing
        t_stats_full = np.zeros(91282)  # Initialize with full brain size
        p_values_full = np.ones(91282)-.00000001
    
        # Only compute statistics for signal vertices
        signal_indices = np.where(signal_mask)[0]

        average_betas_stimuli = np.zeros((91282,tool.shape[1]))
        average_betas_contrast = np.zeros((91282,others.shape[1]))
        logging.info(f"parcel shape: {tool.shape} and {others.shape}")

        count_warnings = 0
        for i,signal_idx in enumerate(signal_indices):
            t_stat, p_val = stats.ttest_ind(tool[i, :], others[i, :],alternative='greater')
            if math.isnan(t_stat) or math.isnan(p_val):
                t_stats_full[signal_idx] = 0
                p_values_full[signal_idx] = 1-.00000001
                warnings.warn(f"t_stat is {t_stat} and p_val is {p_val} at {signal_idx}, {tool[i, :]} > {others[i, :]}",RuntimeWarning)
                count_warnings += 1
            else:
                t_stats_full[signal_idx] = t_stat
                p_values_full[signal_idx] = p_val
            average_betas_stimuli[signal_idx] = tool[i, :]
            average_betas_contrast[signal_idx] = others[i, :]
    
        warnings.warn(f"Saw {count_warnings} warnings", RuntimeWarning)
        binary_mask = np.zeros(91282)
        rejected, corrected_p = fdrcorrection(p_values_full, alpha=0.05)
        binary_mask[np.where((rejected) & (t_stats_full > 0))[0]] = 1

        average_betas_tool = np.average(average_betas_stimuli,axis=1)
        average_betas_others = np.average(average_betas_contrast,axis=1)

        stats_dict = {
            'total_vertices': data.shape[0],
            'signal_mask': signal_mask,
            'signal_vertices': n_signal_vertices,
            't_stats': t_stats_full,
            'p_values': p_values_full,
            'binary_mask': binary_mask,
            'p_values_corrected': corrected_p,
            'betas_stimuli': average_betas_tool,
            'betas_contrast': average_betas_others
        }
        parcel = parcellation if parcellation else 'unparcellated'
        stats_file = os.path.join(output_dir, f"{sub_id}_{task}_stats_{parcel}.npz")
        np.savez(stats_file, **stats_dict)
        logging.info(f"Saved statistics to: {stats_file}")

if __name__ == "__main__":
    load_dotenv()
    base_dir = os.getenv("FRIENDS_BASE_DIR")
    scratch_dir = os.getenv("FRIENDS_SCRATCH_DIR")
    
    parser = ArgumentParser(description="post-processing GLMsingle results")
    parser.add_argument("sub_id", help="Subject ID", type=str)
    parser.add_argument("--task", help="Task", type=str, default="tool")
    parser.add_argument("--parcellation", help="parcellate?",type=str,default=None)
    parser.add_argument("--runs", help="used runs",type=int,default=0)
    args = parser.parse_args()

    base_task = 'wm'
    
    # Set up logging before any other operations
    setup_logging(base_dir, args.sub_id, args.task)
    if args.parcellation:
        parcellation = args.parcellation
    else:
        parcellation = "unparcellated"
    # Continue with the rest of the script...
    result_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/results/{args.sub_id}/{base_task}"
    if bool(args.runs):
        runs = os.listdir(result_dir)
        for run in runs:
            run_result_dir = os.path.join(result_dir,run)
            figure_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/results/{args.sub_id}/{base_task}/{run}/figures"
            glm_results = np.load(os.path.join(run_result_dir, "results.npz"), allow_pickle=True)   
            designinfo = np.load(os.path.join(run_result_dir, "DESIGNINFO.npy"), allow_pickle=True).item()
            stim_files = open(os.path.join(f"{scratch_dir}/hcptrt/output/GLMsingle/", f"{args.sub_id}_{base_task}_unique_predictors.txt"), "r").readlines()
            output_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/mask/{parcellation}/{args.sub_id}/{base_task}/{run}/"

            # Validate input paths
            assert os.path.exists(run_result_dir), f"Results directory not found: {run_result_dir}"
    
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
    
            main(args.sub_id, args.task, glm_results, designinfo, stim_files, output_dir, run,parcellation=args.parcellation)
    else:
        figure_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/results/{args.sub_id}/{base_task}/figures"
        glm_results = np.load(os.path.join(result_dir, "results.npz"), allow_pickle=True)   
        designinfo = np.load(os.path.join(result_dir, "DESIGNINFO.npy"), allow_pickle=True).item()
        stim_files = open(os.path.join(f"{scratch_dir}/hcptrt/output/GLMsingle/", f"{args.sub_id}_{base_task}_unique_predictors.txt"), "r").readlines()
        output_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/mask/{parcellation}/{args.sub_id}/{args.task}/"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
        main(args.sub_id, args.task, glm_results, designinfo, stim_files, output_dir, run=None,parcellation=args.parcellation)

