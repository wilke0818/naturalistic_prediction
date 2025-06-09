import os
import json
import math
from dotenv import load_dotenv
from scipy import stats
from sklearn import metrics
import numpy as np
from argparse import ArgumentParser
from statsmodels.stats.multitest import fdrcorrection
import logging
from datetime import datetime
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import dice
import nibabel as nib

# At the top of the file, after imports
def setup_logging(base_dir, sub_id):
    """Set up logging to both file and console."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(base_dir, 'logs', 'baseline')
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{sub_id}_{timestamp}.log')

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


def pool_dependent_p(p_vals, u_thresh):
    assert u_thresh <= p_vals.shape[0] and u_thresh > 0
    n = p_vals.shape[0]
    results = np.zeros(p_vals.shape[1])
    for j in range(results.shape[0]):
        smallest_sorted_p_vals = sorted(list(p_vals[:,j]))
        adj_p_vals = []
        
        #print(u_thresh, n+1)
        for i in range(1,n-u_thresh+1+1):
            
            adj_p_vals.append(smallest_sorted_p_vals[(u_thresh-1+i)-1]*((n-u_thresh+1)/i))
        #print(adj_p_vals)
        results[j] = min(adj_p_vals)
    return results

def pool_independent_p(p_vals, u_thresh, method='fisher'):
    assert u_thresh <= p_vals.shape[0] and u_thresh > 0
    n = p_vals.shape[0]
    results = np.zeros(p_vals.shape[1])
    for i in range(results.shape[0]):
        largest_sorted_p_vals = sorted(list(p_vals[:,i]),reverse=True)
        #print(largest_sorted_p_vals)
        if method=='fisher':
            statistic = -2 * np.sum(np.log(largest_sorted_p_vals[:n-u_thresh+1]))
            df = 2*(n-u_thresh+1)
            #print(statistic)
            #print(df)
            results[i] = 1 - stats.chi2.cdf(statistic, df)
            if math.isnan(results[i]): print(f"NaN at uthresh {u_thresh}, voxel {i}, statistic was {statistic}, df={df}")
        elif method=='stouffer':
            divisor = n-u_thresh+1
            statistic = np.sum(stats.norm.ppf(1-np.array(largest_sorted_p_vals[0:divisor])))
            results[i] = 1 - stats.norm.cdf(statistic/np.sqrt(divisor))
        else:
            raise ValueError("Method not implemented")
    return results

def pr_auc(y_true, y_pred):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    return metrics.auc(recall, precision)
#test = np.array([[0.5], [0.022], [0.01]])
#print(pool_independent_p(test,1))
#print(pool_independent_p(test,1, method='stouffer'))
#print(pool_independent_p(test,2))
#print(pool_independent_p(test,2, method='stouffer'))
#print(pool_dependent_p(test,1))
#print(pool_dependent_p(test,2))
#print(pool_dependent_p(test,3))

def fdr_pooled_procedure(p_voxel_vals,q=.05):
    sorted_p = sorted(list(p_voxel_vals))
    k = -2
    for j,p_val in enumerate(sorted_p):
        if p_val <= (j+1)/len(sorted_p)*q:
            k = j+1
    corrected_binary = np.zeros(len(sorted_p))
    #print(k)
    #print(k/len(sorted_p)*q)
    corrected_idxs = np.where(p_voxel_vals <= k/len(sorted_p)*q)[0]
    corrected_binary[corrected_idxs] = 1
    #print(p_voxel_vals)
    return corrected_binary

def main(sub_id, results_dir, parcellation=None):
    epsillon = .000000001

    parcellate_str = parcellation if parcellation else 'unparcellated'
    if parcellation == '4s1056':
        #atlas_dlabel = '/home/wilke18/Schaefer2018_400Parcels_17Networks_order_Tian_Subcortex_S4.dlabel.nii'
        atlas_dlabel = '/orcd/data/satra/001/users/jsmentch/atlases/atlas-4S1056Parcels/atlas-4S1056Parcels_space-fsLR_den-91k_dseg.dlabel.nii'
        img = nib.load(atlas_dlabel)
        atlas_data=img.get_fdata()
        atlas_data=atlas_data[0,:]
        atlas_size = int(max(atlas_data))
    elif parcellation=='4s656':
        atlas_dlabel = '/orcd/data/satra/001/users/jsmentch/atlases/atlas-4S656Parcels/atlas-4S656Parcels_space-fsLR_den-91k_dseg.dlabel.nii'
        img = nib.load(atlas_dlabel)
        atlas_data=img.get_fdata()
        atlas_data=atlas_data[0,:]
        atlas_size = int(max(atlas_data))
    elif parcellation=='4s456':
        atlas_dlabel = '/orcd/data/satra/001/users/jsmentch/atlases/atlas-4S456Parcels/atlas-4S456Parcels_space-fsLR_den-91k_dseg.dlabel.nii'
        img = nib.load(atlas_dlabel)
        atlas_data=img.get_fdata()
        atlas_data=atlas_data[0,:]
        atlas_size = int(max(atlas_data))
    elif parcellation=='tian':
        atlas_dlabel = '/home/wilke18/Schaefer2018_400Parcels_17Networks_order_Tian_Subcortex_S4.dlabel.nii'
        img = nib.load(atlas_dlabel)
        atlas_data=img.get_fdata()
        atlas_data=atlas_data[0,:]
        atlas_size = int(max(atlas_data))
    else:
        atlas_size = 91282
    
    subject_results = {}
    results_dir = os.path.join(results_dir,parcellate_str)
    subjects = [directory for directory in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir,directory)) and directory[0:3]=='sub']
    print(subjects)
    for subject in subjects:
        subject_results[subject] = {}
        subject_dir = os.path.join(results_dir,subject)
        tasks = [f for f in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir,f))]
        for task in tasks:
            task_dir = os.path.join(subject_dir, task)
            subject_results[subject][task] = {}
            logging.info(f"Getting results from {subject}'s {task}")
            run_result_dir = os.path.join(task_dir,f"{subject}_{task}_stats_{parcellate_str}.npz")
            run_result = np.load(run_result_dir,allow_pickle=True)
            subject_results[subject][task] = np.ones(atlas_size)
            pos_t_idxs = np.where(run_result['t_stats'] > 0)[0]
            subject_results[subject][task][pos_t_idxs] = run_result['p_values_corrected'][pos_t_idxs]
            subject_results[subject][task] = np.clip(subject_results[subject][task],a_min=epsillon, a_max=1-epsillon)

    logging.info("All results stored in dictionary")

    comparison_methodology = ['dice_coeff', 'roc_auc', 'correlation_coeff']
    subjects = list(subject_results.keys())
    subjects.remove(sub_id)

    all_others = {}
    all_others_z = {}
    ground_truth = {}
    ground_truth_z = {}
    left_out_predictions = {}
    for task in subject_results[sub_id]:
        all_others[task] = []
        all_others_z[task] = []
        left_out_predictions[task] = {}
        ground_truth[task] = subject_results[sub_id][task]
        ground_truth_z[task] = stats.norm.ppf(1-subject_results[sub_id][task])
        print(f"Min and max p ground truth: {np.min(ground_truth[task])} and {np.max(ground_truth[task])}")
        print(f"Min and max z ground truth: {np.min(ground_truth_z[task])} and {np.max(ground_truth_z[task])}")
        binary_ground_truth = np.zeros(ground_truth[task].shape[0])
        binary_ground_truth[np.where(ground_truth[task] < 0.05)[0]] = 1
        for subject in subjects:
            all_others[task].append(subject_results[subject][task])
            all_others_z[task].append(1-stats.norm.ppf(subject_results[subject][task]))
        print(f"Min and max p all_others: {np.min(all_others[task],axis=1)} and {np.max(all_others[task],axis=1)}")
        print(f"Min and max z all_others: {np.min(all_others_z[task],axis=1)} and {np.max(all_others_z[task],axis=1)}")
        # satra method
        satra_method_map = np.clip(1-np.prod(1-np.array(all_others[task]),axis=0),a_min=epsillon,a_max=1-epsillon)
        satra_method_map_z = stats.norm.ppf(1-satra_method_map)
        print(f"Min and max p satra map: {np.min(satra_method_map)} and {np.max(satra_method_map)}")
        print(f"Min and max z satra map: {np.min(satra_method_map_z)} and {np.max(satra_method_map_z)}")
        np.save(os.path.join(results_dir,sub_id,task,f"{sub_id}_{task}_satra_method_{parcellate_str}.npy"),satra_method_map)
        r2 = metrics.r2_score(ground_truth_z[task],satra_method_map_z)
        binary_method_map = np.zeros(satra_method_map.shape[0])
        binary_method_map[np.where(satra_method_map < 0.05)[0]] = 1
        dice_score = 1 - dice(binary_ground_truth,binary_method_map)
        roc_auc = roc_auc_score(binary_ground_truth,binary_method_map)
        left_out_predictions[task]['satra'] = {'dice': dice_score, 'roc-auc': roc_auc, 'r2': r2}

        # p_max
        for u_thresh in range(1,6):
            method_map = np.clip(pool_independent_p(np.array(all_others[task]),u_thresh=u_thresh),a_min=epsillon,a_max=1-epsillon)
            method_map_z = stats.norm.ppf(1-method_map)
            np.save(os.path.join(results_dir,sub_id,task,f"{sub_id}_{task}_p_max_thresh_{u_thresh}_{parcellate_str}.npy"),method_map)
            r2 = metrics.r2_score(ground_truth_z[task],method_map_z)
            binary_method_map = np.zeros(method_map.shape[0])
            binary_method_map[np.where(method_map < 0.05)[0]] = 1
            dice_score = 1 - dice(binary_ground_truth,binary_method_map)
            roc_auc = roc_auc_score(binary_ground_truth,binary_method_map)
            pr_auc_score = pr_auc(binary_ground_truth, binary_method_map)
            
            left_out_predictions[task][f"p_max_thresh_{u_thresh}"] = {'pr_auc': pr_auc_score,'dice': dice_score, 'roc-auc': roc_auc, 'r2': r2}
            binary_method_map_corrected = fdr_pooled_procedure(method_map)
            dice_score = 1 - dice(binary_ground_truth,binary_method_map_corrected)
            roc_auc = roc_auc_score(binary_ground_truth,binary_method_map_corrected)
            pr_auc_score = pr_auc(binary_ground_truth,binary_method_map_corrected)
            left_out_predictions[task][f"p_max_thresh_{u_thresh}_corrected"] = {'pr_auc': pr_auc_score,'dice': dice_score, 'roc-auc': roc_auc, 'r2': r2}


    with open(os.path.join(results_dir,f"{sub_id}_left_out_results_{parcellate_str}.json"),"w") as f:
        json.dump(left_out_predictions,f)

if __name__ == "__main__":
    load_dotenv()
    base_dir = os.getenv("FRIENDS_BASE_DIR")
    scratch_dir = os.getenv("FRIENDS_SCRATCH_DIR")
    
    parser = ArgumentParser(description="get baseline statistics using GLMsingle results")
    parser.add_argument("sub_id", help="Subject ID", type=str)
    parser.add_argument("--parcellation", help="parcellate?", type=str, default=None)
#    parser.add_argument("--task", help="Task", type=str, default="wm")
    args = parser.parse_args()
    
    # Set up logging before any other operations
    setup_logging(base_dir, args.sub_id)
    
    # Continue with the rest of the script...
    result_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/mask/"
    
    main(args.sub_id, result_dir, args.parcellation)
    #runs = os.listdir(result_dir)
    #for run in runs:
    #    run_result_dir = os.path.join(result_dir,run)
    #    figure_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/results/{args.sub_id}/{args.task}/{run}/figures"
    #    glm_results = np.load(os.path.join(run_result_dir, "results.npz"), allow_pickle=True)   
    #    designinfo = np.load(os.path.join(run_result_dir, "DESIGNINFO.npy"), allow_pickle=True).item()
    #    stim_files = open(os.path.join(run_result_dir, f"{args.sub_id}_{args.task}_unique_stim_files.txt"), "r").readlines()
    #    output_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/mask/{args.sub_id}/{args.task}/{run}/"
#
#        # Validate input paths
#        assert os.path.exists(run_result_dir), f"Results directory not found: {run_result_dir}"
#    
#        # Create output directory if it doesn't exist
#        os.makedirs(output_dir, exist_ok=True)
#    
#        main(args.sub_id, args.task, glm_results, designinfo, stim_files, output_dir, run)


