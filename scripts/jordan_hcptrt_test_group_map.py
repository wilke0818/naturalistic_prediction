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

def main(results_dir):

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
            runs =  ['all']
            for run in runs:
                logging.info(f"Getting results from {subject}'s {task} {run}")
                run_result_dir = os.path.join(task_dir,run,f"{subject}_{task}_stats_parcel_binary.npz")
                run_result = np.load(run_result_dir,allow_pickle=True)
                subject_results[subject][task][run] = run_result['p_values']

    logging.info("All results stored in dictionary")


    subjects_maps = {}
    for subject in subject_results:
        subjects_maps[subject] = {}
        for task in subject_results[subject]:
            subjects_maps[subject][task] = {}

            subjects_maps[subject][task]['glm_single'] = subject_results[subject][task]['all']

    task_maps = {}
    for subject in subjects_maps:
        for task in subjects_maps[subject]:
            if task not in task_maps:
                task_maps[task] = []
            task_maps[task].append(subjects_maps[subject][task]['glm_single'])

    for task in task_maps:
        for i in range(1,7):
            fisher = pool_independent_p(np.array(task_maps[task]),u_thresh=i,method='fisher')
            stouffer = pool_independent_p(np.array(task_maps[task]),u_thresh=i,method='stouffer')
            np.save(f"{results_dir}/{task}_glm_single_all_thresh_{i}_fisher.npy", fisher)
            np.save(f"{results_dir}/{task}_glm_single_all_thresh_{i}_stouffer.npy", stouffer)




if __name__ == "__main__":
    load_dotenv()
    base_dir = os.getenv("FRIENDS_BASE_DIR")
    scratch_dir = os.getenv("FRIENDS_SCRATCH_DIR")
    
    
    # Set up logging before any other operations
    
    # Continue with the rest of the script...
    result_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/mask/"
    
    main(result_dir)
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


