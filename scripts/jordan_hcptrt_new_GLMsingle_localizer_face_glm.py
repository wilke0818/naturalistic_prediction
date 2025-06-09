import os
import glob
import pandas as pd
import numpy as np
import json
import time
from argparse import ArgumentParser
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv

import nibabel as nib
import glmsingle
from glmsingle.glmsingle import GLM_single
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import concat_imgs, mean_img, resample_img
from nilearn.glm.regression import OLSModel
from nilearn.glm.contrasts import compute_contrast
from scipy.stats import t

def get_binary_df(df: pd.DataFrame, n_tr: int, sorted_unique_predictors: List[str]) -> pd.DataFrame:
    """
    Create a binary DataFrame representing stimulus occurrences based on predictors.

    Args:
        df (pd.DataFrame): Input DataFrame with onset and predictor information.
        n_tr (int): Number of time points.
        sorted_unique_predictors (List[str]): Sorted list of unique predictor strings.

    Returns:
        pd.DataFrame: Binary DataFrame with TRs as rows and predictor types as columns.
    """
    frame_times = np.arrage(n_tr)
    binary_df = pd.DataFrame(0, index=np.arange(n_tr), columns=sorted_unique_predictors)

    for row in df.itertuples():
        onset = int(row.onset_TR)
        predictor = row.predictor
        if pd.notna(predictor) and predictor in binary_df.columns: # Ensure predictor exists as a column
            binary_df.at[onset, predictor] = 1
    return binary_df



def load_event_and_json_files(event_files: List[str], json_files: List[str], brain_files: List[str], tr: float, sorted_unique_predictors: List[str]) -> List[np.ndarray]:
    """
    Load event and JSON files and create design matrices.

    Args:
        event_files (List[str]): List of event file paths.
        json_files (List[str]): List of JSON file paths.
        brain_files (List[str]): List of brain file paths.
        tr (float): Repetition time.
        sorted_unique_predictors (List[str]): Sorted list of unique predictor strings.

    Returns:
        List[np.ndarray]: List of design matrices.
    """
    design_matrices = []
    for i, (event_file, json_file) in enumerate(zip(event_files, json_files)):
        try:
            with open(json_file, "r") as f:
                json_data = json.load(f)
            try:
                n_tr = len(json_data["time"]["samples"]["AcquisitionNumber"])
            except Exception: # Catch broader exceptions as the specific key might vary
                logging.warning(f"AcquisitionNumber not found in {json_file}, using brain file shape instead")
                n_tr = nib.load(brain_files[i]).shape[0]
            
            df = pd.read_csv(event_file, sep="\t")
            # Drop rows where stim_type is NaN as these won't form valid predictors
            df = df.dropna(subset=["trial_type", "stim_type"])
            #df = df[(df['trial']==1)|(df['trial']=='1')]
            print(df)
            # Create the predictor column
            df["predictor"] = df["trial_type"].astype(str) + "_" + df["stim_type"].astype(str)
            frame_times = np.arange(n_tr) * tr
            events = pd.DataFrame({
                'onset': df['onset'],
                'duration': np.ones(len(df['onset']))*2,
                'trial_type': df['predictor']
                })
            #df["onset_TR"] = df["onset"] / tr
            #binary_df = get_binary_df(df, n_tr, sorted_unique_predictors)
            X = make_first_level_design_matrix(
                frame_times=frame_times,
                events=events,
                hrf_model='glover',  # or 'glover', 'fir', etc.
            )
            design_matrices.append(X)
        except (IOError, json.JSONDecodeError, KeyError, pd.errors.EmptyDataError, FileNotFoundError) as e:
            logging.error(f"Error processing files {event_file} and {json_file}: {str(e)}")
            # Optionally decide if you want to continue or raise the error
            # continue
    return design_matrices

def main(sub_id: str, task: str, tr: float, n_jobs: int, event_files: List[str], json_files: List[str], brain_files: List[str], output_dir: str):
    #event_files = event_files[:5]
    #json_files = json_files[:5]
    #brain_files = brain_files[:5]
    session_ids = [os.path.basename(f).split("_")[1] for f in event_files]
    session_mapping = {session_id: i+1 for i, session_id in enumerate(sorted(set(session_ids)))}
    session_array = np.array([session_mapping[sid] for sid in session_ids])
    print(f"Session array: {session_array}")

    # Collect unique predictors across all runs
    all_unique_predictors = set()
    for event_file in event_files:
        try:
            df = pd.read_csv(event_file, sep="\t")
            # Drop rows only if trial_type is NaN, as it's essential for the predictor
            df = df.dropna(subset=["trial_type", "stim_type"])
            if df.empty:
                logging.warning(f"Skipping file {event_file} as it has no rows with valid trial_type.")
                continue

            # Create predictor column conditionally
            df["predictor"] = df["trial_type"].astype(str) + "_" + df["stim_type"].astype(str)
            all_unique_predictors.update(df["predictor"].unique())
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            logging.warning(f"Skipping file {event_file} due to error: {e}")
            continue

    # Sort the unique predictors to ensure consistent order
    sorted_unique_predictors = sorted(all_unique_predictors)
    logging.info(f"Found {len(sorted_unique_predictors)} unique predictors.")

    # Use the sorted_unique_predictors when loading data
    design_matrices = load_event_and_json_files(event_files, json_files, brain_files, tr, sorted_unique_predictors)

    brain_data = []
    for brain_file in brain_files:
        try:
            img = nib.load(brain_file)
            brain_data.append(img.get_fdata())
        except Exception as e:
            logging.error(f"Error loading or reshaping brain file {brain_file}: {str(e)}")
            # Depending on requirements, you might want to skip this run or raise the error
            # For now, let's raise to halt execution if a brain file fails
            raise

    # Check consistency after potentially skipping runs in data loading
    if not design_matrices or not brain_data:
         logging.error("No valid design matrices or brain data could be loaded. Exiting.")
         return # Or raise an exception

    # Ensure counts match AFTER loading attempts (some might have been skipped)
    if len(design_matrices) != len(brain_data):
        logging.error(f"Mismatch after loading: {len(design_matrices)} design matrices vs {len(brain_data)} brain data sets.")
        # Decide how to handle mismatch: maybe only use common runs? For now, error out.
        raise ValueError("Mismatch between successfully loaded design matrices and brain data.")

    # Adjust session_array if runs were skipped during loading (This part is tricky and needs careful implementation)
    # Simplest approach: If runs are skipped, we might need to re-evaluate which sessions are present.
    # For now, we assume load_event_and_json_files and the brain_data loop either succeed completely or fail early.
    # A more robust implementation would track indices of successfully loaded runs.
    assert len(design_matrices) == len(session_array), \
        f"Number of design matrices ({len(design_matrices)}) and session array ({len(session_array)}) do not match after potential skips."


    # save unique predictors to a file
    predictor_file_path = os.path.join(output_dir, f"{sub_id}_{task}_unique_predictors.txt")
    with open(predictor_file_path, "w") as f:
        for predictor in sorted_unique_predictors:
            f.write(f"{predictor}\n")
    logging.info(f"Saved unique predictors to {predictor_file_path}")

    stimdur = 2
    start_time = time.time()

    results_dir = os.path.join(output_dir, f"results3/{sub_id}/{task}")
    os.makedirs(results_dir, exist_ok=True)
    figure_dir = os.path.join(results_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    # repetition time, in seconds
    #t_r = 2.0
    # Sample at the beginning of each acquisition.
    slice_time_ref = 0.0
    # We use a discrete cosine transform to model signal drifts.
    drift_model = "Cosine"
    # The cutoff for the drift model is 0.01 Hz.
    high_pass = 0.01
    # The hemodynamic response function
    hrf_model = "glover"
    #opt = {
    #    'wantlibrary': 1,
    #    'wantglmdenoise': 1,
    #    'wantfracridge': 1,
    #    'wantfileoutputs': [0,0,0,0],
    #    'wantmemoryoutputs': [1,1,1,1],
    #    "sessionindicator": session_array,
    #    'n_jobs': n_jobs
    #}
    #glmsingle_obj = GLM_single(opt)
    #print(f'running GLMsingle...')
    #try:
    #    results = glmsingle_obj.fit(design_matrices, brain_data, stimdur, tr, outputdir=results_dir, figuredir=figure_dir)
    #except OSError as e:
    #    logging.error(f"Encountered error during GLMsingle fitting: {str(e)}")
    #    raise
    glms = []
    contrasts = {
        'body': np.array([1.0/2,-1.0/6,-1.0/6,-1.0/6,1.0/2,-1.0/6,-1.0/6,-1.0/6]),
        'face': np.array([-1.0/6,1.0/2,-1.0/6,-1.0/6,-1.0/6,1.0/2,-1.0/6,-1.0/6]),
        'place': np.array([-1.0/6,-1.0/6,1.0/2,-1.0/6,-1.0/6,-1.0/6,1.0/2,-1.0/6]),
        'tool': np.array([-1.0/6,-1.0/6,-1.0/6,1.0/2,-1.0/6,-1.0/6,-1.0/6,1.0/2]),
        'wm': np.array([-1.0/4,-1.0/4,-1.0/4,-1.0/4,1.0/4,1.0/4,1.0/4,1.0/4])
    }
    for design_matrix, brain_response in zip(design_matrices, brain_data):
        print(design_matrix)
        betas = []
        residuals = []
        sigma_squared = []
       
        contrast_results = []
        for v in range(brain_response.shape[1]):
            contrast_result = {}
            model = OLSModel(design_matrix)
            
            y = brain_response[:, v]
            results = model.fit(y)
            beta = results.theta
            res = results.residuals
#            sig2 = results.sigma2
            #beta, res, rank, s = ols(y, design_matrix)
            betas.append(beta)
            residuals.append(res)
            for contrast in contrasts:
                contrast_vector = np.pad(contrasts[contrast],(0,design_matrix.shape[1]-len(contrasts[contrast])))
                contrast_beta = contrast_vector @ beta
                contrast_result[contrast] = contrast_beta
            contrast_results.append(contrast_result)

#            sigma_squared.append(sig2)  # residual variance
        
        glms.append({
            'betas': betas,
            'residuals': residuals,
            'contrasts': contrast_results
#            'sigma_squared': sigma_squared
            })
    #glms = FirstLevelModel(
    #    t_r=tr,
    #    slice_time_ref=slice_time_ref,
    #    hrf_model="glover",
    #).fit(run_imgs=brain_data, events=design_matrices)
    all_betas = [glm['betas'] for glm in glms]
    beta_results_file = os.path.join(results_dir, f"results_betas.npz")
    np.savez(beta_results_file, **{'betas': all_betas, 'sessions': session_array})
    design = pd.get_dummies(session_array,drop_first=True)
    design['intercept'] = 1

    for contrast in contrasts:
        t_vals = []
        p_vals = []
        contrast_vals = []
        betas = []
        X = design.values
        X = design.values.astype(np.float64)
        #pinv_X = np.linalg.pinv(X)
        XtX_inv = np.linalg.inv(X.T @ X)
        contrast_vec = np.zeros(X.shape[1])
        contrast_vec[-1] = 1
        #contrast_data = np.array([glm['contrasts'][contrast] for glm in glms])
        #print(contrast_data.shape)
        print(contrast_vec.shape)
        n_runs = X.shape[0]
        n_regressors = X.shape[1]
        for v in range(91282):#contrast_data.shape[1]):
            contrast_data = np.array([glm['contrasts'][v][contrast] for glm in glms])
            y = contrast_data
            model = OLSModel(X)
            results = model.fit(y)
            beta = results.theta
            residuals = results.residuals
            # Manually compute sigma²
            sigma2 = np.sum(residuals ** 2) / (n_runs - n_regressors)
            se = np.sqrt(sigma2 * (contrast_vec @ XtX_inv @ contrast_vec))
            contrast_val = contrast_vec @ beta
            t_stat = contrast_val/se
            dof = n_runs - n_regressors
            p_val = 2 * (1 - t.cdf(np.abs(t_stat), df=dof))
            t_vals.append(t_stat)
            p_vals.append(p_val)
            betas.append(beta)
            contrast_vals.append(contrast_val)
        print(contrast)
        print(t_vals)
        print(p_vals)
        print(contrast_vals)
        print(betas)
        results = {
                't_stats': t_vals,
                'p_vals': p_vals,
                'betas': betas,
                'contrast_vals': contrast_vals
                }
        
        
    # save results to a file
        results_file = os.path.join(results_dir, f"results_{contrast}.npz")
        np.savez(results_file, **results)
        print(f"Saved results for session to {results_file}")
    
    elapsed_time = time.time() - start_time
    print(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    base_dir = os.getenv("FRIENDS_BASE_DIR")
    scratch_dir = os.getenv("FRIENDS_SCRATCH_DIR")
    dataset_dir = os.getenv("FRIENDS_DATASET_DIR")
    glm_data_dir = os.getenv("FRIENDS_DATA_DIR")

    if not all([base_dir, scratch_dir, dataset_dir]):
        raise ValueError("Missing required environment variables")

    data_dir = Path(dataset_dir) / "fmriprep" / "hcptrt"
    glm_dir = Path(glm_data_dir) / "cneuromod_postproc" / "hcptrt_cleaned_globalz"

    parser = ArgumentParser()
    parser.add_argument("sub_id", type=str)  # This remains required as a positional argument
    parser.add_argument("--task", type=str, default="wm")  # Optional with default
    parser.add_argument("--tr", type=float, default=1.49)  # Optional with default
    parser.add_argument("--n_jobs", type=int, default=32)  # Optional with default
    args = parser.parse_args()

    event_files = sorted(data_dir.glob(f"sourcedata/hcptrt/{args.sub_id}/*/func/*{args.task}*_events.tsv"))
    json_files = sorted(data_dir.glob(f"sourcedata/hcptrt/{args.sub_id}/*/func/*{args.task}*_bold.json"))
    #sub-01_ses-010_task-wm_run-2_space-fsLR_den-91k_bold_cleaned.dtseries.nii
    brain_files = sorted(glm_dir.glob(f"{args.sub_id}/*_task-{args.task}*tseries.nii"))

    output_dir = os.path.join(scratch_dir, f"hcptrt_glm/output/GLMsingle")
    os.makedirs(output_dir, exist_ok=True)

    if not all([event_files, json_files, brain_files]):
        raise FileNotFoundError("Required files not found")

    logging.info(f"Found {len(event_files)} event files, {len(json_files)} JSON files, and {len(brain_files)} brain files")

    main(args.sub_id, args.task, args.tr, args.n_jobs, event_files, json_files, brain_files, output_dir)
