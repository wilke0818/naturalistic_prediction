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

    subject_results = {}
    subjects = [directory for directory in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir,directory))]
    print(subjects)
    for subject in subjects:
        subject_results[subject] = {}
        subject_dir = os.path.join(results_dir,subject)
        tasks = os.listdir(subject_dir)
        for task in tasks:
            task_dir = os.path.join(subject_dir, task)
            subject_results[subject][task] = {}
            runs = os.listdir(task_dir)
            for run in runs:
                logging.info(f"Getting results from {subject}'s {task} {run}")
                run_result_dir = os.path.join(task_dir,run,f"{subject}_{task}_stats_zero_noise.npz")
                run_result = np.load(run_result_dir,allow_pickle=True)
                subject_results[subject][task][run] = run_result['p_values']

    logging.info("All results stored in dictionary")

    comparison_methodology = ['explained_variance', 'corr']
    tasks = list(subject_results[sub_id].keys())

    metric_results = {}
    subjects_maps = {}
    for subject in subject_results:
        metric_results[subject] = {}
        subjects_maps[subject] = {}
        for task in subject_results[subject]:
            metric_results[subject][task] = {'average': {}, 't_min': {}}
            subjects_maps[subject][task] = {'t_min': {}}
            runs = list(subject_results[subject][task].keys())
            runs.remove('all')
            subject_task_map = []
            for run in runs:
                metric_results[subject][task][run] = {}
                subject_task_map.append(subject_results[subject][task][run])
                for comp in comparison_methodology:
                    if comp == 'explained_variance':
                        #print('all',subject_results[subject][task]['all']['p'].shape, subject_results[subject][task]['all']['error'])
                        #print(subject,task,run,subject_results[subject][task][run]['p'].shape, subject_results[subject][task][run]['error'])
                        #nan_mask = np.isnan(subject_results[subject][task]['all']['p'])

                        # To get the indices of NaN values:
                        #nan_indices = np.where(nan_mask)[0]
                        #print(nan_indices.shape)
                        #breaking=False
                        #if nan_indices.shape[0]>0: breaking=True
                        #nan_mask = np.isnan(subject_results[subject][task][run]['p'])

                        # To get the indices of NaN values:
                        #nan_indices = np.where(nan_mask)[0]
                        #print(nan_indices.shape)
                        #if nan_indices.shape[0]>0 or breaking: break
                        metric_results[subject][task][run][comp] = metrics.explained_variance_score(subject_results[subject][task]['all'],subject_results[subject][task][run])
                    elif comp == 'corr':
                        metric_results[subject][task][run][comp] = stats.pearsonr(subject_results[subject][task]['all'],subject_results[subject][task][run]).statistic
            u_threshs = [1, math.ceil(.25*len(subject_task_map)), math.ceil(.5*len(subject_task_map)), math.ceil(.75*len(subject_task_map)), len(subject_task_map)]
            for u_thresh in u_threshs:
                subjects_maps[subject][task]['t_min'][u_thresh] = pool_dependent_p(np.array(subject_task_map), u_thresh)
                metric_results[subject][task]['t_min'][u_thresh] = {}

            subjects_maps[subject][task]['average'] = np.average(np.array(subject_task_map),axis=0)
            logging.info(f"Average mask shape for {subject}'s {task} is {subjects_maps[subject][task]['average'].shape}")

            for comp in comparison_methodology:
                if comp == 'explained_variance':
                    metric_results[subject][task]['average'][comp] = metrics.explained_variance_score(subject_results[subject][task]['all'], subjects_maps[subject][task]['average'])
                elif comp == 'corr':
                    metric_results[subject][task]['average'][comp] = stats.pearsonr(subject_results[subject][task]['all'],subjects_maps[subject][task]['average'])
                for thresh in u_threshs:
                    if comp == 'explained_variance':
                        metric_results[subject][task]['t_min'][thresh][comp] = metrics.explained_variance_score(subject_results[subject][task]['all'], subjects_maps[subject][task]['t_min'][thresh])
                    elif comp == 'corr':
                        metric_results[subject][task]['t_min'][thresh][comp] = stats.pearsonr(subject_results[subject][task]['all'],subjects_maps[subject][task]['t_min'][thresh])
    
    with open(os.path.join(results_dir,f"map_analysis_zero_noise.json"),"w") as f:
        json.dump(metric_results, f)

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


