import os
import json
from typing import Any
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import itertools
import seaborn as sns
import pickle

from models_optimized import HierarchicalPlasticRNN
from task import WhatAndWhereTask

def generate_perturbation_vector(all_models):
    """
    Generate perturbation vectors for each model.
    For each model, generates one perturbation direction aligned with the block type readout
    
    Args:
        all_models: List of trained HierarchicalPlasticRNN models
    
    Returns:
        perturbation_vectors: List of perturbation vectors, where each element is a tensor of shape (hidden_dim, )
    """
    aligned_perturbation_vectors = []
    # orthogonal_perturbation_vectors = []
    
    for model in all_models:
        # Extract block type readout weights (hidden_size * num_areas, 2)
        block_type_weights = model.h2o['block_type'].effective_weight().detach()
        
        # Get the readout direction (first component of the 2D output)
        block_type_direction = block_type_weights[1]-block_type_weights[0]  # Shape: (hidden_size)
        
        # Normalize the block type direction
        block_type_direction = block_type_direction / torch.linalg.norm(block_type_direction)
        
        aligned_perturbation_vectors.append(block_type_direction)
    
    aligned_perturbation_vectors = torch.stack(aligned_perturbation_vectors) # (num_models, hidden_dim)
    
    return aligned_perturbation_vectors


def test_model_with_perturbation(all_models, perturbation_vectors, args, ortho_control=False, test_samples=10):
    """
    Perturbation of the activity of the model, testing the effect on behavior.
    For each block, we keep the hidden state across trials as if it were not perturbed. 
    For each trial of each block, we add the perturbation and record only the subsequent choice.
    
    Args:
        all_models: List of trained models
        perturbation_vectors: List of lists, where each inner list contains perturbation directions for each model
        args: Arguments for the task
        ortho_control: Whether to use the perturbation vector, or randomly permute the perturbation vector every block
        test_samples: Number of test samples per condition
    Returns:
        results_dict: Dictionary containing model results with input perturbation
    """
    for model in all_models:
        model.eval()

    mdl_indices = list(range(len(all_models))) # 8 models
    block_type_indices = list(range(2)) # 2 block types: where and what
    reversal_interval_indices = [30, 35, 40, 45, 50] # 5 reversal intervals: 30, 35, 40, 45, 50
    initial_better_option_indices = [0, 1] # 2 initial better options: L/R, A/B
    sample_indices = list(range(test_samples)) # test samples per condition
    perturbation_strengths = [-0.2, -0.1, 0, 0.1, 0.2]
    
    indices_product = list(itertools.product(mdl_indices, block_type_indices, 
                                                                        reversal_interval_indices, 
                                                                        initial_better_option_indices, 
                                                                        sample_indices))
    
    # calculate the sizes of array dimensions for saving results
    num_models = len(all_models)
    num_block_types = len(block_type_indices)
    num_reversal_intervals = len(reversal_interval_indices)
    num_initial_better_options = len(initial_better_option_indices)
    num_sample_indices = len(sample_indices)
    num_blocks = num_models * num_block_types * num_reversal_intervals * num_initial_better_options * num_sample_indices
    
    num_trials = args['trials_per_test_block']
    
    batch_size = args['batch_size']
    assert batch_size == 1, 'batch size must be 1 for this test'
    
    num_perturb_conditions = len(perturbation_strengths)
    num_timesteps = what_where_task.T_ITI + what_where_task.T_fixation + what_where_task.T_stim + what_where_task.T_choice_reward
    timestep_windows = np.cumsum([0, what_where_task.T_ITI, what_where_task.T_fixation, 
                                  what_where_task.T_stim, what_where_task.T_choice_reward]).astype(int)
    
    stim_dims = args['stim_dims']
    hidden_dim = args['hidden_size']
    
    # preallocate numpy arrays for saving results
    results_dict = {
        'model_index':           np.full((num_blocks,), np.nan),
        'reversal_interval':     np.full((num_blocks,), np.nan),
        'block_type':            np.full((num_blocks,), np.nan),
        'initial_better_option': np.full((num_blocks,), np.nan),
        'stimulus':              np.full((num_blocks, num_trials), np.nan),
        'inputs':                np.full((num_blocks, num_trials, 2*stim_dims), np.nan),
        'reward_probs':          np.full((num_blocks, num_trials, 2), np.nan),
        'img_chosen':            np.full((num_blocks, num_perturb_conditions, num_trials), np.nan),
        'loc_chosen':            np.full((num_blocks, num_perturb_conditions, num_trials), np.nan),
        'reward':                np.full((num_blocks, num_perturb_conditions, num_trials), np.nan),
        'perturbation_strength': np.full((num_blocks, num_perturb_conditions), np.nan),
        'neuron_state':          np.full((num_blocks, num_perturb_conditions, num_trials, num_timesteps, hidden_dim), np.nan),
        'synaptic_state':        np.full((num_blocks, num_perturb_conditions, num_trials, hidden_dim, hidden_dim), np.nan),
    }

    device = torch.device('cpu')

    # loop through all conditions
    with torch.no_grad():
        for block_idx, indices in enumerate(indices_product):
            
            mdl_idx, test_block_type, test_reversal_interval, test_initial_better_option, sample_idx = indices

            print(f'Testing model {mdl_idx+1} out of {num_models} '
                f'with block type {test_block_type+1} out of {num_block_types} '
                f'and reversal interval {test_reversal_interval+1} out of {num_reversal_intervals} '
                f'and initial better option {test_initial_better_option+1} out of {num_initial_better_options} '
                f'and sample index {sample_idx+1} out of {num_sample_indices}')

            
            # load perturbation direction, either aligned with or randomly permuted
            if not ortho_control:
                current_perturbation_direction = perturbation_vectors[mdl_idx]
            else:
                current_perturbation_direction = perturbation_vectors[mdl_idx][torch.randperm(perturbation_vectors[mdl_idx].shape[0])]
            
            # generate trials for the current condition, use for all perturbation conditions
            trial_info = what_where_task.generate_trials(
                batch_size = args['batch_size'],
                trials_per_block = args['trials_per_test_block'], 
                reversal_interval = [test_reversal_interval, test_reversal_interval],
                reward_schedule=[args['reward_probs_high'], 1-args['reward_probs_high']],
                block_type=test_block_type,
                initial_better_option=test_initial_better_option,
            ) 

            # save results for the current condition
            results_dict['model_index'][block_idx] = mdl_idx
            results_dict['reversal_interval'][block_idx] = test_reversal_interval
            results_dict['block_type'][block_idx] = test_block_type
            results_dict['initial_better_option'][block_idx] = test_initial_better_option
            
            results_dict['stimulus'][block_idx] = trial_info['stim_configs'].squeeze(1)
            results_dict['inputs'][block_idx] = trial_info['stim_inputs'].squeeze(1)
            results_dict['reward_probs'][block_idx] = trial_info['reward_probs'].squeeze(1)            
                
            for pert_idx, pert_strength in enumerate(perturbation_strengths):
                print(f'Testing perturbation strength {pert_strength} out of {num_perturb_conditions}')
                current_perturbation = current_perturbation_direction * pert_strength
                results_dict['perturbation_strength'][block_idx, pert_idx] = pert_strength

                # use the same sequence of stimuli for all perturbation conditions
                stim_inputs = trial_info['stim_inputs'].to(device, dtype=torch.float)
                rewards = trial_info['rewards'].to(device)
                
                # initialize hidden state for the current model
                hidden, w_hidden = all_models[mdl_idx].init_hidden(batch_size=args['batch_size'], device=device)

                # for each trial, test effect of perturbation on subsequent choice
                for trial_idx in range(len(stim_inputs)):
                    ''' first phase, give nothing '''
                    all_x = {
                        'go_cue': torch.zeros(args['batch_size'], 1, device=device),
                        'fixation': torch.zeros(args['batch_size'], 1, device=device),
                        'stimulus': torch.zeros_like(stim_inputs[trial_idx]),
                        # 'reward': torch.zeros(args['batch_size'], 2, device=device), # 2+2 for chosen and unchosen rewards
                        'action_chosen': torch.zeros(args['batch_size'], 2, device=device), # left/right  
                    }

                    _, hidden, w_hidden, hs = all_models[mdl_idx](all_x, steps=what_where_task.T_ITI, 
                                                    neumann_order=0,
                                                    hidden=hidden, w_hidden=w_hidden, 
                                                    DAs=torch.zeros(args['batch_size'], device=device), 
                                                    save_all_states=False,
                                                    current_perturbation=current_perturbation)

                    results_dict['neuron_state'][block_idx, pert_idx, trial_idx, timestep_windows[0]:timestep_windows[1]] = hs
                    
                    ''' second phase, give fixation '''
                    all_x = {
                        'go_cue': torch.ones(args['batch_size'], 1, device=device),
                        'fixation': torch.ones(args['batch_size'], 1, device=device),
                        'stimulus': torch.zeros_like(stim_inputs[trial_idx]),
                        # 'reward': torch.zeros(args['batch_size'], 2, device=device), # 2+2 for chosen and unchosen rewards
                        'action_chosen': torch.zeros(args['batch_size'], 2, device=device), # left/right
                    }

                    _, hidden, w_hidden, hs = all_models[mdl_idx](all_x, steps=what_where_task.T_fixation, 
                                                    neumann_order=0,
                                                    hidden=hidden, w_hidden=w_hidden, 
                                                    DAs=torch.zeros(args['batch_size'], device=device), 
                                                    save_all_states=False,
                                                    current_perturbation=current_perturbation)

                    results_dict['neuron_state'][block_idx, pert_idx, trial_idx, timestep_windows[1]:timestep_windows[2]] = hs

                    ''' third phase, give stimuli and no feedback '''
                    all_x = {
                        'go_cue': torch.ones(args['batch_size'], 1, device=device),
                        'fixation': torch.ones(args['batch_size'], 1, device=device),
                        'stimulus': stim_inputs[trial_idx],  # Use perturbed stimulus inputs
                        # 'reward': torch.zeros(args['batch_size'], 2, device=device), # 2+2 for chosen and unchosen rewards
                        'action_chosen': torch.zeros(args['batch_size'], 2, device=device), # left/right
                    }

                    output, hidden, w_hidden, hs = all_models[mdl_idx](all_x, steps=what_where_task.T_stim, 
                                                        neumann_order=0,
                                                        hidden=hidden, w_hidden=w_hidden, 
                                                        DAs=torch.zeros(args['batch_size'], device=device), 
                                                        save_all_states=False,
                                                        current_perturbation=current_perturbation)
                    
                    results_dict['neuron_state'][block_idx, pert_idx, trial_idx, timestep_windows[2]:timestep_windows[3]] = hs

                    ''' use output to calculate action, reward, and record loss function '''
                    action = torch.multinomial(output['action'].softmax(-1), num_samples=1).squeeze(-1) # (batch size, )
                    rwd_ch = rewards[trial_idx][torch.arange(args['batch_size']),action] # (batch size, )

                    results_dict['img_chosen'][block_idx, pert_idx, trial_idx] = (action!=trial_info['stim_configs'][trial_idx])*1
                    # (loc_choice, config)->img_chosen, (0,0)->0, (1,0)->1, (0,1)->1, (1,1)->0
                    results_dict['loc_chosen'][block_idx, pert_idx, trial_idx] = action.squeeze(0)
                    results_dict['reward'][block_idx, pert_idx, trial_idx] = rwd_ch.squeeze(0)

                    '''fourth phase, give stimuli and choice, and update weights'''
                    all_x = {
                        'go_cue': torch.ones(args['batch_size'], 1, device=device),
                        'fixation': torch.ones(args['batch_size'], 1, device=device),
                        'stimulus': stim_inputs[trial_idx-1],  # Use perturbed stimulus inputs here too
                        # 'reward': torch.eye(2, device=device)[None][torch.arange(args['batch_size']), rwd_ch], # reward/reward
                        'action_chosen': torch.eye(2, device=device)[None][torch.arange(args['batch_size']), action], # left/right
                    }

                    output, hidden, w_hidden, _ = all_models[mdl_idx](all_x, steps=what_where_task.T_choice_reward, 
                                                    neumann_order=0,
                                                    hidden=hidden, w_hidden=w_hidden, 
                                                    DAs=(2*rwd_ch-1), save_all_states=False,
                                                    current_perturbation=current_perturbation)

                    results_dict['neuron_state'][block_idx, pert_idx, trial_idx, timestep_windows[3]:timestep_windows[4]] = hs
                    results_dict['synaptic_state'][block_idx, pert_idx, trial_idx, :, :] = w_hidden

    for k, v in results_dict.items():
        print(k, v.shape)

    return results_dict


# Example usage:
if __name__ == "__main__":
    exp_dir = '/dartfs-hpc/rc/home/d/f005d7d/attn-rnn/what_where_analysis/what-where-task-analysis/rnn/exp'
    figure_data_dir = '/dartfs-hpc/rc/home/d/f005d7d/attn-rnn/what_where_analysis/what-where-task-analysis/figures'

    model_array_dir = [f'delay_mse_dv_loss_{i}' for i in range(1,9)]
    # model_array_dir = [f'test{i}' for i in range(1,9)]

    f = open(os.path.join(exp_dir, model_array_dir[0], 'args.json'), 'r')
    args = json.load(f)
    print('loaded args')

    what_where_task = WhatAndWhereTask(args['dt'], args['stim_dims'])

    input_config = {
        'go_cue': (1, [0]),
        'fixation': (1, [0]),
        'stimulus': (args['stim_dims']*2, [0]),
        # 'reward': (2, [0]), 
        'action_chosen': (2, [0]), 
    }

    output_config = {
        'action': (2, [0], True), # left, right
        'stimulus': (args['stim_dims'], [0], True),
        'block_type': (2, [0], True), # where or what block
        'dv_loc': (2, [0], True), # desired location based on previous trial outcome
        'dv_stim': (2, [0], True), # location of desired stimulus based on previous trial outcome
    }

    total_trial_time = what_where_task.times['ITI']+\
                        what_where_task.times['fixation_time']+\
                        what_where_task.times['stim_time']+\
                        what_where_task.times['choice_reward_time']

    model_specs = {'input_config': input_config, 'hidden_size': args['hidden_size'], 'output_config': output_config,
                    'num_areas': args['num_areas'], 'plastic': args['plas_type']=='all', 'activation': args['activ_func'],
                    'dt_x': args['dt'], 'dt_w': total_trial_time, 'tau_x': args['tau_x'], 'tau_w': args['tau_w'], 
                    'e_prop': args['e_prop'], 'init_spectral': args['init_spectral'], 'balance_ei': args['balance_ei'],
                    'sigma_rec': args['sigma_rec'], 'sigma_in': args['sigma_in'], 'sigma_w': args['sigma_w'], 
                    'inter_regional_sparsity': (1, 1), 'inter_regional_gain': (1, 1)}

    device = torch.device('cpu')
    E_SIZE = int(args['hidden_size']*args['e_prop'])

    all_models = []
    for model_dir in model_array_dir:
        model = HierarchicalPlasticRNN(**model_specs)
        state_dict = torch.load(os.path.join(exp_dir, model_dir, 'checkpoint.pth.tar'), 
                                map_location=torch.device('cpu'))['model_state_dict']
        print(model.load_state_dict(state_dict))
        all_models.append(model)
        print(f'model at {model_dir} loaded successfully')

    # Generate perturbation vectors for each model
    print("Generating perturbation vectors...")
    perturbation_vectors = generate_perturbation_vector(all_models)
    
    test_activities_dir = '/dartfs/rc/lab/S/SoltaniA/f005d7d/what_where_analysis/rnn_test_activities/test_activities_with_perturbations.pkl'

    all_saved_states = {'aligned_perturbation': [], 'orthogonal_perturbation': []}

    print('testing perturbation aligned with block type readout')
    all_saved_states['aligned_perturbation'] = \
        test_model_with_perturbation(all_models, args=args, test_samples=1, perturbation_vectors=perturbation_vectors, ortho_control=False)

    print('testing perturbation orthogonal to block type readout')
    all_saved_states['orthogonal_perturbation'] = \
        test_model_with_perturbation(all_models, args=args, test_samples=1, perturbation_vectors=perturbation_vectors, ortho_control=True)

    print('simulation complete')
    with open(test_activities_dir, 'wb') as f:
        pickle.dump(all_saved_states, f)
    print(f'saved results to {test_activities_dir}')
