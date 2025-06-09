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

#test = np.array([[0.5], [0.022], [0.01]])
#print(pool_independent_p(test,1))
#print(pool_independent_p(test,1, method='stouffer'))
#print(pool_independent_p(test,2))
#print(pool_independent_p(test,2, method='stouffer'))
#print(pool_dependent_p(test,1))
#print(pool_dependent_p(test,2))
#print(pool_dependent_p(test,3))

def main(sub_id, results_dir):
    atlas_dlabel = '/home/wilke18/Schaefer2018_400Parcels_17Networks_order_Tian_Subcortex_S4.dlabel.nii'
    img = nib.load(atlas_dlabel)
    atlas_data=img.get_fdata()
    atlas_data=atlas_data[0,:]
    atlas_size = int(max(atlas_data))
    
    subject_results = {}
    subjects = [directory for directory in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir,directory)) and directory[0:3]=='sub']
    print(subjects)
    for subject in subjects:
        subject_results[subject] = {}
        subject_dir = os.path.join(results_dir,subject)
        tasks = [f for f in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir,f))]
        for task in tasks:
            task_dir = os.path.join(subject_dir, task)
            subject_results[subject][task] = {}
            runs =  [f for f in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir,f))]
            for run in runs:
                logging.info(f"Getting results from {subject}'s {task} {run}")
                run_result_dir = os.path.join(task_dir,run,f"{subject}_{task}_stats_parcel_binary.npz")
                run_result = np.load(run_result_dir,allow_pickle=True)
                subject_results[subject][task][run] = run_result['p_values']

    logging.info("All results stored in dictionary")

    comparison_methodology = ['dice_coeff', 'roc_auc']
    tasks = list(subject_results[sub_id].keys())

    #metric_results = {}
    subjects_maps = {}
    for subject in subject_results:
        subjects_maps[subject] = {}
        for task in subject_results[subject]:
            subjects_maps[subject][task] = {'t_min': {}}
            runs = list(subject_results[subject][task].keys())
            runs.remove('all')
            subject_task_map = []
            for run in runs:
                #metric_results[subject][task][run] = {}
                subject_task_map.append(subject_results[subject][task][run])

                     #   metric_results[subject][task][run][comp] = stats.pearsonr(subject_results[subject][task]['all'],subject_results[subject][task][run]).statistic
            u_threshs = [1, math.ceil(.25*len(subject_task_map)), math.ceil(.5*len(subject_task_map)), math.ceil(.75*len(subject_task_map)), len(subject_task_map)]
            for u_thresh in u_threshs:
                subjects_maps[subject][task]['t_min'][u_thresh] = pool_dependent_p(np.array(subject_task_map), u_thresh)
                #metric_results[subject][task]['t_min'][u_thresh] = {}

                temp = np.zeros(atlas_size)
                temp[np.where(fdrcorrection(subjects_maps[subject][task]['t_min'][u_thresh],alpha=0.05)[0])[0]] = 1
                #print(subject, task,u_thresh,temp)
                print(fdrcorrection(subjects_maps[subject][task]['t_min'][u_thresh],alpha=0.05)[1])
                print(fdrcorrection(subjects_maps[subject][task]['t_min'][u_thresh])[1])
                #print(u_thresh,subjects_maps[subject][task]['t_min'][u_thresh])
                np.save(f"{results_dir}/{subject}/{task}/t-min-{u_thresh}_binary_mask.npy",temp)

            subjects_maps[subject][task]['satra'] = 1-(np.prod(1-np.array(subject_task_map),axis=0))#np.average(np.array(subject_task_map),axis=0)
            temp = np.zeros(atlas_size)
            temp[np.where(fdrcorrection(subjects_maps[subject][task]['satra'],alpha=0.05)[0])[0]] = 1
            #print(subjects_maps[subject][task]['satra']-np.average(np.array(subject_task_map),axis=0))
            np.save(f"{results_dir}/{subject}/{task}/satra_mask_binary.npy",temp)
            subjects_maps[subject][task]['glm_single'] = subject_results[subject][task]['all']
            logging.info(f"Average mask shape for {subject}'s {task} is {subjects_maps[subject][task]['satra'].shape}")

    #with open(os.path.join(results_dir,f"map_analysis.json"),"w") as f:
    #    json.dump(metric_results, f)
    subjects = list(subjects_maps.keys())
    subjects.remove(sub_id)

    left_out_predictions = {}
    
    for task in subjects_maps[sub_id]:
        left_out_predictions[task] = {}
        for map_type in subjects_maps[sub_id][task]:
            
            #left_out_predictions[task][map_type] = []
            for subject in subjects:
                if map_type!='t_min':
                    if map_type not in left_out_predictions[task]:
                        left_out_predictions[task][map_type] = []
                    left_out_predictions[task][map_type].append(subjects_maps[subject][task][map_type])
                else:
                    t_mins = sorted(list(subjects_maps[subject][task][map_type]))
                    for i in range(len(t_mins)):
                        map_type_key = f't_min_{i}'
                        if map_type_key not in left_out_predictions[task]:
                            left_out_predictions[task][map_type_key] = []
                        left_out_predictions[task][f't_min_{i}'].append(subjects_maps[subject][task][map_type][t_mins[i]])

    left_out_prediction_maps = {}
    for task in left_out_predictions:
        left_out_prediction_maps[task] = {}
        for map_type in left_out_predictions[task]:
            left_out_prediction_maps[task][map_type] = {}
            left_out_prediction_maps[task][map_type]['satra'] = {}
            left_out_avg_map_p = 1-np.prod(1-np.array(left_out_predictions[task][map_type]),axis=0) #np.average(np.array(left_out_predictions[task][map_type]),axis=0)
            left_out_avg_map = np.zeros(atlas_size)
            left_out_avg_map[np.where(fdrcorrection(left_out_avg_map_p, alpha=0.05)[0])[0]] = 1
            np.save(f"{results_dir}/{sub_id}/{task}/{map_type}_satra_mask_binary.npy",left_out_avg_map)
            logging.info(f"Got average map for left out {sub_id} on task {task} when other subjects were combined with {map_type}")
            for left_out_ground_truth_map_type in subjects_maps[sub_id][task]:
                if left_out_ground_truth_map_type!='t_min':
                    temp = np.zeros(atlas_size)
                    temp[np.where(fdrcorrection(subjects_maps[sub_id][task][left_out_ground_truth_map_type])[0])[0]] = 1
                    dice_score = dice(temp,left_out_avg_map)
                    dice_score = 0 if math.isnan(dice_score) else 1-dice_score
                    roc = roc_auc_score(temp,left_out_avg_map, average='weighted')
                    roc = 0 if math.isnan(roc) else roc
                    left_out_prediction_maps[task][map_type]['satra'][left_out_ground_truth_map_type] = {'dice': dice_score, 'auc-roc': roc}
                else:
                    for t_min in subjects_maps[sub_id][task]['t_min']:
                        temp = np.zeros(atlas_size)
                        temp[np.where(fdrcorrection(subjects_maps[sub_id][task][left_out_ground_truth_map_type][t_min])[0])[0]] = 1
                        dice_score = dice(temp,left_out_avg_map)
                        dice_score = 0 if math.isnan(dice_score) else 1-dice_score
                        roc = roc_auc_score(temp,left_out_avg_map, average='weighted')
                        roc = 0 if math.isnan(roc) else roc
                        left_out_prediction_maps[task][map_type]['satra'][f"t_min_{t_min}"] = {'dice': dice_score, 'auc-roc': roc}
            t_mins = list(range(1,len(left_out_predictions[task][map_type])+1))
            runs = list(subject_results[sub_id][task].keys())
            runs.remove('all')
            ind_runs = []
            for run in runs:
                temp = np.zeros(atlas_size)
                temp[np.where(fdrcorrection(subject_results[sub_id][task][run])[0])[0]] = 1
                dice_score = dice(temp,left_out_avg_map)
                dice_score = 0 if math.isnan(dice_score) else 1-dice_score
                roc = roc_auc_score(temp,left_out_avg_map, average='weighted')
                roc = 0 if math.isnan(roc) else roc
                ind_runs.append([dice_score, roc])
            left_out_prediction_maps[task][map_type]['satra']['individual'] = {'dice': np.average(ind_runs,axis=0)[0], 'auc-roc':np.average(ind_runs,axis=0)[1]}
            
            for t_min in t_mins:
                left_out_prediction_maps[task][map_type][f't_min_{t_min}'] = {}
                left_out_t_min_map_p = pool_independent_p(np.array(left_out_predictions[task][map_type]),t_min)
                left_out_t_min_map = np.zeros(atlas_size)
                left_out_t_min_map[np.where(fdrcorrection(left_out_t_min_map_p)[0])[0]] = 1
                np.save(f"{results_dir}/{sub_id}/{task}/{map_type}_t-min-{t_min}_binary_mask.npy",left_out_t_min_map)
                #nan_mask = np.isnan(left_out_t_min_map)
                #nan_indices = np.where(nan_mask)[0]
                #if len(nan_indices) != 0:
                #    logging.info(f"task {task} on {map_type} with t_min_{t_min} has nan")
                #    continue
                ind_runs = []
                for run in runs:
                    temp = np.zeros(atlas_size)
                    temp[np.where(fdrcorrection(subject_results[sub_id][task][run])[0])[0]] = 1
                    dice_score = dice(temp,left_out_t_min_map)
                    dice_score = 0 if math.isnan(dice_score) else 1-dice_score
                    roc = roc_auc_score(temp,left_out_t_min_map, average='weighted')
                    roc = 0 if math.isnan(roc) else roc
                    ind_runs.append([dice_score, roc])
                left_out_prediction_maps[task][map_type][f't_min_{t_min}']['individual'] = {'dice': np.average(ind_runs,axis=0)[0], 'auc-roc':np.average(ind_runs,axis=0)[1]}
                for left_out_ground_truth_map_type in subjects_maps[sub_id][task]:
                    if left_out_ground_truth_map_type!='t_min':
                        temp = np.zeros(atlas_size)
                        temp[np.where(fdrcorrection(subjects_maps[sub_id][task][left_out_ground_truth_map_type])[0])[0]] = 1
                        dice_score = dice(temp,left_out_t_min_map)
                        dice_score = 0 if math.isnan(dice_score) else 1-dice_score
                        roc = roc_auc_score(temp,left_out_t_min_map, average='weighted')
                        roc = 0 if math.isnan(roc) else roc
                        left_out_prediction_maps[task][map_type][f't_min_{t_min}'][left_out_ground_truth_map_type] = {'dice': dice_score, 'auc-roc': roc}
                    else:
                        for left_out_t_min in subjects_maps[sub_id][task]['t_min']:
                            temp = np.zeros(atlas_size)
                            temp[np.where(fdrcorrection(subjects_maps[sub_id][task][left_out_ground_truth_map_type][left_out_t_min])[0])[0]] = 1
                            dice_score = dice(temp,left_out_t_min_map)
                            dice_score = 0 if math.isnan(dice_score) else 1-dice_score
                            roc = roc_auc_score(temp,left_out_t_min_map, average='weighted')
                            roc = 0 if math.isnan(roc) else roc
                            left_out_prediction_maps[task][map_type][f't_min_{t_min}'][f"t_min_{left_out_t_min}"] = {'dice': dice_score, 'auc-roc': roc}

    with open(os.path.join(results_dir,f"{sub_id}_left_out_results.json"),"w") as f:
        json.dump(left_out_prediction_maps,f)

if __name__ == "__main__":
    load_dotenv()
    base_dir = os.getenv("FRIENDS_BASE_DIR")
    scratch_dir = os.getenv("FRIENDS_SCRATCH_DIR")
    
    parser = ArgumentParser(description="get baseline statistics using GLMsingle results")
    parser.add_argument("sub_id", help="Subject ID", type=str)
#    parser.add_argument("--task", help="Task", type=str, default="wm")
    args = parser.parse_args()
    
    # Set up logging before any other operations
    setup_logging(base_dir, args.sub_id)
    
    # Continue with the rest of the script...
    result_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/mask/"
    
    main(args.sub_id, result_dir)
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


