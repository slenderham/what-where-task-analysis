import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
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


def process_unit_regression(unit_idx, unit_firing_rates, df_template, formula, contrasts_to_test, num_timesteps):
    """
    Process regression for a single unit across all time points.
    Module-level function to avoid pickling large arrays in closure.
    
    Args:
        unit_idx: Index of the unit
        unit_firing_rates: Array of shape (num_trials, num_timesteps) - only this unit's data
        df_template: DataFrame template (will be copied)
        formula: Regression formula string
        contrasts_to_test: Array of contrasts
        num_timesteps: Number of time steps
    
    Returns:
        (unit_idx, unit_results) tuple
    """
    # Create copy of template for this unit
    df_template_unit = df_template.copy()
    df_template_unit['fr'] = 0  # Initialize column, will be updated each time point
    
    unit_results = []
    for time_idx in range(num_timesteps):
        # Update 'fr' column instead of creating new DataFrame
        df_template_unit['fr'] = unit_firing_rates[:, time_idx]
        
        # fit linear model
        mdl = smf.ols(formula, df_template_unit).fit()
        contr_pvals = np.array(mdl.t_test(contrasts_to_test).pvalue)
        
        # calculate anova
        anova_mdl = sm.stats.anova_lm(mdl, typ=3)

        # calculate effect size for anova
        ms_error = anova_mdl.loc[:,'sum_sq'].iloc[-1]/anova_mdl.loc[:,'df'].iloc[-1]
        omega_sq = (anova_mdl.loc[:,'sum_sq'].iloc[:-1]-anova_mdl.loc[:,'df'].iloc[:-1]*ms_error)/ \
                        (anova_mdl.loc[:,'sum_sq'].sum()+ms_error)

        pvals = anova_mdl.loc[:,'PR(>F)'].iloc[:-1].to_numpy().squeeze()

        unit_results.append([mdl.params, omega_sq, pvals, contr_pvals, mdl.aic])
    
    return unit_idx, unit_results


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
    stride = int(0.01/binsize)

    all_sess_regression_info = {
        'aligned_event': [],
        'monkey_name': [],
        'area_name': [],
        'sess_date': [],
        'betas': [],
        'exp_vars': [],
        'pvals': [],
        'aics': [],
        'contr_pvals': []
    }

    # 12 regressors, separate by blocks
    regressor_names = ['type', 'block',
                       'S_curr', 'C_what_curr', 'C_where_curr', 'R_curr',
                       'C_what_prev', 'C_where_prev', 'R_prev',
                       'RXC_where', 'RXC_what',
                       'SXC_what', 'RXS', 'RXSXC_what']

    regressor_expr = ['C(block_type, Sum)/C(block_id, Sum)',
                       '(C(block_type, Sum)/C(block_id, Sum))*(C_what_curr:C_where_curr)',
                       '(C(block_type, Sum)/C(block_id, Sum))*C_what_curr',
                       'C(block_type, Sum)*C_where_curr', 'C(block_type, Sum)*R_curr',
                       '(C(block_type, Sum)/C(block_id, Sum))*C_what_prev',
                       'C(block_type, Sum)*C_where_prev', 'C(block_type, Sum)*R_prev',
                       'C(block_type, Sum)*(R_prev:C_where_prev)',
                       '(C(block_type, Sum)/C(block_id, Sum))*(R_prev:C_what_prev)',
                       'C(block_type, Sum)*(C_what_curr:C_where_curr:C_what_prev)',
                       '(C(block_type, Sum)/C(block_id, Sum))*(R_prev:C_what_curr:C_where_curr)',
                       'C(block_type, Sum)*(R_prev:C_what_curr:C_where_curr:C_what_prev)']


    var_names_in_table = ['C_what_curr', 'C_where_curr', 'R_curr', 'block_type', 'block_id',
                          'C_what_prev', 'C_where_prev', 'R_prev']

    formula = 'fr~'+'+'.join(regressor_expr)

    contrasts_to_test = []
    base_inds = [72, 100, 104, 130, 156]

    num_betas = 158
    num_exp_vars = 32

    for base_ind in base_inds:
        curr_contr_what = np.zeros(num_betas)
        curr_contr_what[base_ind] = 1
        curr_contr_what[base_ind+1] = 1
        contrasts_to_test.append(curr_contr_what)

        curr_contr_where = np.zeros(num_betas)
        curr_contr_where[base_ind] = 1
        curr_contr_where[base_ind+1] = -1
        contrasts_to_test.append(curr_contr_where)

    contrasts_to_test = np.stack(contrasts_to_test)
    num_contrs = contrasts_to_test.shape[0]

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
                task_info_prev = np.concatenate([np.array([[0, 0, 0]]), task_info[:-1, :3]], axis=0)

                # put together design matrix
                # C_what_curr, C_where_curr, R_curr, block_id, block_type
                # C_what_prev, C_where_prev, R_curr
                X = np.concatenate([task_info, task_info_prev], axis=1)

                # Quick Win #1: Pre-build DataFrame template once (outside all loops)
                df_template = pd.DataFrame(X, columns=var_names_in_table)

                all_units_beta = np.ones((num_timesteps, num_units, num_betas))*np.nan
                all_units_exp_var = np.ones((num_timesteps, num_units, num_exp_vars))*np.nan
                all_units_pvals = np.ones((num_timesteps, num_units, num_exp_vars))*np.nan
                all_units_contr_pvals = np.ones((num_timesteps, num_units, num_contrs))*np.nan
                all_units_aics = np.ones((num_timesteps, num_units))*np.nan

                # Quick Win #2: Process entire unit at once, parallelize across units
                # Use module-level function to avoid pickling large arrays (neural_data) in closure
                # Instead, pass only the unit's firing rate data (much smaller)
                linreg_anova_results = Parallel(n_jobs=args.njobs, timeout=999999)(
                    delayed(process_unit_regression)(
                        unit_idx, 
                        neural_data[:, unit_idx, :],  # Pass only this unit's data, not entire neural_data array
                        df_template,
                        formula,
                        contrasts_to_test,
                        num_timesteps
                    ) for unit_idx in tqdm(range(num_units), desc="Processing units"))

                # Unpack results
                for unit_idx, unit_results in linreg_anova_results:
                    all_units_beta[:, unit_idx, :] = np.stack([curr_time_results[0] for curr_time_results in unit_results])
                    all_units_exp_var[:, unit_idx, :] = np.stack([curr_time_results[1] for curr_time_results in unit_results])
                    all_units_pvals[:, unit_idx, :] = np.stack([curr_time_results[2] for curr_time_results in unit_results])
                    all_units_contr_pvals[:, unit_idx, :] = np.stack([curr_time_results[3] for curr_time_results in unit_results])
                    all_units_aics[:, unit_idx] = np.stack([curr_time_results[4] for curr_time_results in unit_results])

                    '''
                    bootstrapping for the null distribution of frac. significant cells and correlation between coefficients
                    for each unit, shuffle within each block N times
                    then sample 1000 combinations to get random populations
                    '''

                # all_sess_regression_info['neural_data'].append(neural_data)
                all_sess_regression_info['monkey_name'].append(monkey_name)
                all_sess_regression_info['aligned_event'].append(aligned_event)
                all_sess_regression_info['area_name'].append(area_idx)
                all_sess_regression_info['sess_date'].append(sess_date)
                all_sess_regression_info['betas'].append(all_units_beta)
                all_sess_regression_info['exp_vars'].append(all_units_exp_var)
                all_sess_regression_info['pvals'].append(all_units_pvals)
                all_sess_regression_info['contr_pvals'].append(all_units_contr_pvals)
                all_sess_regression_info['aics'].append(all_units_aics)


                with open(os.path.join(processed_path, 'all_sess_regression_info_small_stride.pkl'), 'wb') as f:
                    pickle.dump(all_sess_regression_info, f)