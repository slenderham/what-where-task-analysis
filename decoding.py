import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import glob
import h5py
import re
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve
from tqdm import tqdm
import scipy.io as sio
from joblib import Parallel, delayed
import os
import pickle
import argparse


def run_decoding_cross_time(train_time_idx, curr_cond_neural_data, curr_label, num_timesteps, num_resamples):
    """
    Run decoding for one training time point, testing on all time points.
    Module-level function to avoid pickling large arrays in closure.
    
    Args:
        train_time_idx: Training time point index
        neural_data: Array of shape (num_trials, num_units, num_timesteps)
        curr_test_data_selector: Boolean array for trial selection
        curr_label: Array of labels for selected trials
        num_timesteps: Number of time steps
        num_resamples: Number of resampling iterations
    
    Returns:
        [all_ws, all_dec_accs] list
    """
    # Extract training data once
    curr_time_fr_train = curr_cond_neural_data[:, :, train_time_idx]  # (num_trials, num_units) for training
    
    # fit decoding for CV
    all_dec_accs = np.nan*np.empty((num_timesteps, num_resamples))  # num_timesteps X num_resamples
    for sample_idx in range(num_resamples):
        kf = KFold(n_splits=10)
        curr_sample_dec_accs = np.nan*np.empty((10, num_timesteps))
        for fold_idx, (train_index, test_index) in enumerate(kf.split(curr_time_fr_train)):
            clf_cv = LogisticRegression(C=10**6)
            clf_cv = clf_cv.fit(curr_time_fr_train[train_index], curr_label[train_index])
            
            for test_time_idx in range(num_timesteps):
                curr_time_fr_test = curr_cond_neural_data[:, :, test_time_idx]
                curr_sample_dec_accs[fold_idx, test_time_idx] = clf_cv.score(curr_time_fr_test[test_index], curr_label[test_index])

        all_dec_accs[:, sample_idx] = np.mean(curr_sample_dec_accs, axis=0)

    # fit decoding for all
    clf_full = LogisticRegression(C=10**6).fit(curr_time_fr_train, curr_label)
    assert(len(clf_full.classes_)==2)

    all_ws = np.concatenate([clf_full.intercept_, clf_full.coef_.squeeze()])

    return [all_ws, all_dec_accs]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--njobs", default=8, help="number of cores to use")
    args = parser.parse_args()

    monkey_names = ["W", "V"]
    # aligned_events = ["StimOnset", "Choice", "RewFeedback"]
    aligned_events = ["StimOnset"]

    # root_dir = '/Users/f005d7d/Documents/Attn_MdPRL/what-where-task/'
    root_dir = '/dartfs/rc/lab/S/SoltaniA/f005d7d/what_where_analysis/'

    bhv_path = os.path.join(root_dir, 'Behavior/')
    processed_path = os.path.join(root_dir, 'processed/')
    neural_path = os.path.join(root_dir, 'RasterVec_binSize_10ms/')

    binsize = 0.01
    # gauss_SD = 0.02/binsize
    win_size = int(0.05/binsize)
    stride = int(0.05/binsize)

    num_resamples = 100

    all_sess_decoding_info = {
        'aligned_event': [],
        'monkey_name': [],
        'area_name': [],
        'sess_date': [],
        'ws': [],
        'accs': []
    }

    for event_idx, aligned_event in enumerate(aligned_events):
        for monkey_idx, monkey_name in enumerate(monkey_names):
            files = glob.glob(
                f'{neural_path}/{aligned_event}/RastVect-{monkey_name}*-binsize10ms-align2{aligned_event}.mat')
            for sess_idx in range(len(files)):
                filename = files[sess_idx]

                curr_sess_neural = h5py.File(filename)
                sess_date = re.search(re.compile(
                    f'RastVect-{monkey_name}(\\d*)-binsize10ms-align2{aligned_event}.mat'), filename).groups()[0]

                # neural_data = gaussian_filter(
                #     curr_sess_neural['aligned2event'], gauss_SD, mode='constant', axes=2)
                neural_data = convolve(curr_sess_neural['aligned2event'], np.ones((1,1,win_size))/win_size, mode='valid')[:,:,::stride]

                bhv_filename = bhv_path+'SPKcounts_'+monkey_name+sess_date+'cue_MW_250X250ms.mat'
                curr_sess_bhv = sio.loadmat(
                    bhv_path+'SPKcounts_'+monkey_name+sess_date+'cue_MW_250X250ms.mat')
                task_info = curr_sess_bhv['Y']

                # only keep chosen image, chosen loc, reward, block type, block id
                task_info = task_info[:, [0, 1, 2, 9, 7]].astype(float)

                trial_mask = task_info[:, 4] <= 24
                task_info = task_info[trial_mask]
                neural_data = neural_data[trial_mask]
                area_idx = curr_sess_neural['vectorInfo']['Array'][:].squeeze()

                neuron_mask = np.nonzero(np.min(neural_data.sum(0), 1))[0]
                neural_data = neural_data[:, neuron_mask, :]
                area_idx = area_idx[neuron_mask].squeeze()

                num_trials, num_units, num_timesteps = neural_data.shape

                print("--------------------------------------------------------------------")
                print("aligned to: " + aligned_event + ", monkey: " +
                      monkey_name + ", session: " + sess_date + ", #trials=" + str(num_trials))

                task_info[:, :3] = task_info[:, :3]*2-1
                task_info[:, 3] = task_info[:, 3]*2-3
                task_info[:, 4] = task_info[:, 4]%12

                # make the time-lagged part of the design
                task_info_prev = task_info[:-1, :3]

                # put together design matrix
                # C_what_curr, C_where_curr, R_curr, block_id, block_type
                # C_what_prev, C_where_prev, R_prev
                all_task_info = np.concatenate([task_info[1:], task_info_prev], axis=1)
                neural_data = neural_data[1:]

                unit_idxs = np.abs(neural_data).mean((0,2)) < 0.5
                neural_data = neural_data[:, unit_idxs, :]
                print(f"discarded units: {np.nonzero(~unit_idxs)}")
                num_units = neural_data.shape[1]

                area_idx = area_idx[unit_idxs]

                '''
                for each decoding test, get 
                (1) data selectors for choosing which rows to choose
                (2) label encoder for getting the output class
                '''

                '''
                block_type and R_prev
                '''
                data_selectors = [
                    lambda x: (x[:, 3]==-1) & (x[:, 7]==-1), lambda x: (x[:, 3]==1) & (x[:, 7]==-1),
                    lambda x: (x[:, 3]==-1) & (x[:, 7]==1), lambda x: (x[:, 3]==1) & (x[:, 7]==1),
                    lambda x: (x[:, 3]==-1) & (x[:, 7]==-1), lambda x: (x[:, 3]==1) & (x[:, 7]==-1),
                    lambda x: (x[:, 3]==-1) & (x[:, 7]==1), lambda x: (x[:, 3]==1) & (x[:, 7]==1),
                    lambda x: (x[:, 3]==-1) & (x[:, 7]==-1), lambda x: (x[:, 3]==1) & (x[:, 7]==-1),
                    lambda x: (x[:, 3]==-1) & (x[:, 7]==1), lambda x: (x[:, 3]==1) & (x[:, 7]==1),
                ]

                '''
                C_where_curr, C_where_prev, SXC_what_prev
                '''
                label_encoder = [
                    lambda x: x[:, 1], lambda x: x[:, 1], 
                    lambda x: x[:, 1], lambda x: x[:, 1],
                    lambda x: x[:, 6], lambda x: x[:, 6], 
                    lambda x: x[:, 6], lambda x: x[:, 6], 
                    lambda x: x[:, 0]*x[:, 1]*x[:, 5], lambda x: x[:, 0]*x[:, 1]*x[:, 5], 
                    lambda x: x[:, 0]*x[:, 1]*x[:, 5], lambda x: x[:, 0]*x[:, 1]*x[:, 5],
                ]

                num_dec_tests = len(data_selectors)

                all_test_ws = np.ones((num_timesteps, num_dec_tests, num_units+1))*np.nan
                all_test_accs = np.ones((num_timesteps, num_dec_tests, num_timesteps, num_resamples))*np.nan

                for idx_dec_test in tqdm(range(num_dec_tests), desc="Decoding tests"):
                    
                    # (num_trials,), mask for trials in the current condition
                    curr_test_data_selector = data_selectors[idx_dec_test](all_task_info) 
                    # (num_trials,), masked labels for the current condition
                    curr_label = label_encoder[idx_dec_test](all_task_info)[curr_test_data_selector]
                    # (num_selected_trials, num_units, num_timesteps)
                    curr_cond_neural_data = neural_data[curr_test_data_selector]

                    decoding_results = Parallel(n_jobs=args.njobs, verbose=10)(
                        delayed(run_decoding_cross_time)(
                            train_time_idx,
                            curr_cond_neural_data,
                            curr_label,
                            num_timesteps,
                            num_resamples
                        ) for train_time_idx in range(num_timesteps))

                    all_test_ws[:, idx_dec_test, :] = np.stack([curr_time_results[0] for curr_time_results in decoding_results])
                    all_test_accs[:, idx_dec_test, :, :] = np.stack([curr_time_results[1] for curr_time_results in decoding_results])

                # all_sess_regression_info['neural_data'].append(neural_data)
                all_sess_decoding_info['monkey_name'].append(monkey_name)
                all_sess_decoding_info['aligned_event'].append(aligned_event)
                all_sess_decoding_info['area_name'].append(area_idx)
                all_sess_decoding_info['sess_date'].append(sess_date)
                all_sess_decoding_info['ws'].append(all_test_ws)
                all_sess_decoding_info['accs'].append(all_test_accs)

                with open(os.path.join(processed_path, 'all_sess_decoding_info.pkl'), 'wb') as f:
                    pickle.dump(all_sess_decoding_info, f)