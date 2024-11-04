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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--njobs", default=8, help="number of cores to use")
    args = parser.parse_args()

    monkey_names = ["W", "V"]
    # aligned_events = ["StimOnset", "Choice", "RewFeedback"]
    aligned_events = ["StimOnset"]

    # root_dir = '/Users/f005d7d/Documents/Attn_MdPRL/what-where-task/'
    root_dir = '/dartfs-hpc/scratch/f005d7d/what_where_analysis/'

    bhv_path = os.path.join(root_dir, 'Behavior/')
    processed_path = os.path.join(root_dir, 'processed/')
    neural_path = os.path.join(root_dir, 'RasterVec_binSize_10ms/')

    binsize = 0.01
    # gauss_SD = 0.02/binsize
    win_size = int(0.05/binsize)
    stride = int(0.05/binsize)

    all_sess_stability_decoding_info = {
        'aligned_event': [],
        'monkey_name': [],
        'area_name': [],
        'sess_date': [],
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

                if monkey_idx==0 and sess_idx==2:
                    unit_idxs = np.ones((neural_data.shape[1]))
                    unit_idxs[572] = 0
                    neural_data = neural_data[:,unit_idxs>0.5]
                    num_units -= 1

                if monkey_idx==1 and sess_idx==3:
                    unit_idxs = np.ones((neural_data.shape[1]))
                    unit_idxs[429] = 0
                    neural_data = neural_data[:,unit_idxs>0.5]
                    num_units -= 1

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
                    lambda x: (x[:, 3]==-1) & (x[:, 7]==1), lambda x: (x[:, 3]==1) & (x[:, 7]==1)
                ]

                '''
                train on C_where_curr
                '''
                label_encoder = [
                    lambda x: x[:, 1], lambda x: x[:, 6], lambda x: x[:, 0]*x[:, 1]*x[:, 5]
                ]

                num_labels = len(label_encoder)
                num_data_trains = len(data_selectors)
                num_data_tests = len(data_selectors)-1

                all_test_accs = np.ones((num_timesteps, num_labels, num_data_trains, num_data_tests))*np.nan

                for idx_label in tqdm(range(num_labels)):
                    for idx_data_train in range(num_data_trains):
                        
                        ''' train on one variable in one set of conditions'''
                        curr_train_data_selector = data_selectors[idx_data_train](all_task_info)
                        curr_train_label = label_encoder[idx_label](all_task_info)[curr_train_data_selector]

                        def run_decoding(time_idx):
                            # select one condition for fitting choice decoder
                            curr_time_fr = neural_data[:, :, time_idx]
                            curr_time_train_fr = curr_time_fr[curr_train_data_selector]

                            clf = SGDClassifier(alpha=1e-6, loss='hinge', max_iter=1000, tol=1e-4)
                            clf = clf.fit(curr_time_train_fr, curr_train_label)
                    
                            curr_time_dec_accs = []

                            for idx_data_test in range(num_data_trains):
                                if idx_data_train==idx_data_test:
                                    continue

                                curr_test_data_selector = data_selectors[idx_data_test](all_task_info)
                                curr_test_label = label_encoder[idx_label](all_task_info)[curr_test_data_selector]

                                curr_time_test_fr = curr_time_fr[curr_test_data_selector]

                                curr_time_dec_accs.append(clf.score(curr_time_test_fr, curr_test_label))

                            curr_time_dec_accs = np.array(curr_time_dec_accs) # num_data_tests

                            return curr_time_dec_accs

                        decoding_results = Parallel(n_jobs=args.njobs)(
                            delayed(run_decoding)(time_idx) for time_idx in range(num_timesteps))

                        all_test_accs[:, idx_label, idx_data_train, :] = np.stack([curr_time_results for curr_time_results in decoding_results])

                # all_sess_regression_info['neural_data'].append(neural_data)
                all_sess_stability_decoding_info['monkey_name'].append(monkey_name)
                all_sess_stability_decoding_info['aligned_event'].append(aligned_event)
                all_sess_stability_decoding_info['area_name'].append(area_idx)
                all_sess_stability_decoding_info['sess_date'].append(sess_date)
                all_sess_stability_decoding_info['accs'].append(all_test_accs)

                with open(os.path.join(processed_path, 'all_sess_stability_decoding_info.pkl'), 'wb') as f:
                    pickle.dump(all_sess_stability_decoding_info, f)