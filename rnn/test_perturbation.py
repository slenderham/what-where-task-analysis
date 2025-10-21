import os
import json
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

def generate_perturbation_vector(all_models, n_orthogonal_directions=5):
    """
    Generate perturbation vectors for each model.
    For each model, generates:
    1. One perturbation direction aligned with the block type readout
    N. N orthogonal directions perpendicular to the block type readout
    
    Args:
        all_models: List of trained HierarchicalPlasticRNN models
        n_orthogonal_directions: Number of orthogonal directions to generate
    
    Returns:
        perturbation_vectors: List of lists, where each inner list contains:
                             [block_type_direction, orthogonal_direction_1, ..., orthogonal_direction_N]
    """
    aligned_perturbation_vectors = []
    orthogonal_perturbation_vectors = []
    
    for model_idx, model in enumerate(all_models):
        # Extract block type readout weights (hidden_size * num_areas, 2)
        block_type_weights = model.h2o['block_type'].effective_weight().detach()
        
        # Get the readout direction (first component of the 2D output)
        block_type_direction = block_type_weights[1]-block_type_weights[0]  # Shape: (hidden_size)
        
        # Normalize the block type direction
        block_type_direction = block_type_direction / torch.linalg.norm(block_type_direction)
        
        # Generate orthogonal directions using Gram-Schmidt process
        hidden_dim = block_type_direction.shape[0]
        
        # Use QR decomposition to efficiently construct the orthogonal directions
        # First, construct an initial basis with the block type direction
        basis = block_type_direction.reshape(-1,1)  # Shape: (hidden_dim, 1)
        # Add (n_orthogonal_directions) random full-rank vectors to the basis
        random_matrix = np.random.randn(hidden_dim, n_orthogonal_directions)
        full_matrix = np.concatenate([basis, random_matrix], axis=1)  # (hidden_dim, n_orthogonal_directions + 1)
        # Perform QR decomposition
        Q, _ = np.linalg.qr(full_matrix)
        # The first column of Q is aligned with block_type_direction (up to sign),
        # the next n_orthogonal_directions columns are orthogonal to it and to each other
        orthogonal_directions = torch.from_numpy(Q[:, 1:]).float().t() # (n_orthogonal_directions, hidden_dim)
        
        # Combine all directions for this model
        aligned_perturbation_vectors.append(block_type_direction)
        orthogonal_perturbation_vectors.append(orthogonal_directions)

    aligned_perturbation_vectors = torch.stack(aligned_perturbation_vectors) # (num_models, hidden_dim)
    orthogonal_perturbation_vectors = torch.stack(orthogonal_perturbation_vectors) # (num_models, n_orthogonal_directions, hidden_dim)
    
    return aligned_perturbation_vectors, orthogonal_perturbation_vectors


def test_model_with_input_perturbation(all_models, perturbation_vectors, test_samples=10):
    """
    Alternative version that adds perturbation directly to input vectors instead of hidden states.
    
    Args:
        all_models: List of trained models
        test_samples: Number of test samples per condition
        perturbation_vectors: List of lists, where each inner list contains perturbation directions for each model
    Returns:
        results_dict: Dictionary containing model results with input perturbation
    """
    for model in all_models:
        model.eval()

    results_dict = {
        'model_index': [], # num_blocks
        'perturbation_index': [], # num_blocks - tracks which perturbation direction was applied
        'reversal_interval': [], # num_blocks
        'block_type': [], # num_blocks
        'stimulus': [], # num_blocks X num_trials
        'inputs': [], # num_blocks X num_trials X 2*num_dims
        'img_chosen': [], # num_blocks X num_trials
        'loc_chosen': [], # num_blocks X num_trials
        'reward': [], # num_blocks X num_trials
        'perturbation_strength': [], # num_blocks
        'perturbation_phase': [], # num_blocks
        'reward_probs': [], # num_blocks X num_trials X 2
        'neuron_states': [], # num_blocks X num_trials X num_timesteps X num_units
        'synaptic_states': [], # num_blocks X num_trials X num_units X num_units
    }

    mdl_indices = list(range(len(all_models)))
    block_type_indices = list(range(2))
    reversal_interval_indices = [30, 35, 40, 45, 50]

    perturbation_strengths = [-1, -0.5, 0.5, 1]
    perturbation_phases = [1, 2, 3]
    indices_product = list(itertools.product(mdl_indices, block_type_indices, reversal_interval_indices))
    pert_indices = list(itertools.product(perturbation_strengths, perturbation_phases))

    # Load args to get model parameters
    exp_dir = '/dartfs-hpc/rc/home/d/f005d7d/attn-rnn/what_where_analysis/what-where-task-analysis/rnn/exp'
    model_array_dir = [f'simple_setup_{i}' for i in range(1,9)]
    f = open(os.path.join(exp_dir, model_array_dir[0], 'args.json'), 'r')
    args = json.load(f)
    
    what_where_task = WhatAndWhereTask(args['dt'], args['stim_dims'])
    device = torch.device('cpu')

    with torch.no_grad():
        for indices in indices_product:
            mdl_idx, test_block_type, test_reversal_interval = indices
            current_perturbation_direction = perturbation_vectors[mdl_idx]
            print(f'Testing model {mdl_idx} out of {len(all_models)} '
                  f'with block type {test_block_type} out of {len(block_type_indices)} '
                  f'and reversal interval {test_reversal_interval} out of {len(reversal_interval_indices)}')

            for _ in range(test_samples):
                trial_info = what_where_task.generate_trials(
                    batch_size = args['batch_size'],
                    trials_per_block = args['trials_per_test_block'], 
                    reversal_interval = [test_reversal_interval, test_reversal_interval],
                    reward_schedule=[args['reward_probs_high'], 1-args['reward_probs_high']],
                    block_type=test_block_type,
                ) 
                
                # use the same sequence of stimuli for all perturbation conditions
                for pert_idx in pert_indices:
                    perturbation_strength, perturbation_phase = pert_idx
                    current_perturbation = current_perturbation_direction * perturbation_strength
                    stim_inputs = trial_info['stim_inputs'].to(device, dtype=torch.float)
                    rewards = trial_info['rewards'].to(device)
                    
                    hidden = None
                    w_hidden = None

                    results_dict['model_index'].append(mdl_idx)
                    results_dict['perturbation_index'].append(pert_idx)
                    results_dict['perturbation_strength'].append(perturbation_strength)
                    results_dict['perturbation_phase'].append(perturbation_phase)

                    results_dict['reversal_interval'].append(test_reversal_interval)
                    results_dict['block_type'].append(test_block_type)
                    results_dict['stimulus'].append(trial_info['stim_configs'])
                    results_dict['inputs'].append(trial_info['stim_inputs'])
                    results_dict['reward_probs'].append(trial_info['reward_probs'])

                    results_dict['img_chosen'].append([])
                    results_dict['loc_chosen'].append([])
                    results_dict['reward'].append([])
                    
                    results_dict['neuron_states'].append([])
                    results_dict['synaptic_states'].append([])
                    
                    for i in range(len(stim_inputs)):
                        results_dict['neuron_states'][-1].append([])
                        
                        ''' first phase, give nothing '''
                        all_x = {
                            'fixation': torch.zeros(args['batch_size'], 1, device=device),
                            'stimulus': torch.zeros_like(stim_inputs[i]),
                            'reward': torch.zeros(args['batch_size'], 2, device=device), # 2+2 for chosen and unchosen rewards
                            'action_chosen': torch.zeros(args['batch_size'], 2, device=device), # left/right  
                        }

                        _, hidden, w_hidden, hs = all_models[mdl_idx](all_x, steps=what_where_task.T_ITI, 
                                                        neumann_order=0,
                                                        hidden=hidden, w_hidden=w_hidden, 
                                                        DAs=None, save_all_states=True,
                                                        perturbation=None)
                        results_dict['neuron_states'][-1][-1].append(hs)
                        
                        ''' second phase, give fixation '''
                        all_x = {
                            'fixation': torch.ones(args['batch_size'], 1, device=device),
                            'stimulus': torch.zeros_like(stim_inputs[i]),
                            'reward': torch.zeros(args['batch_size'], 2, device=device), # 2+2 for chosen and unchosen rewards
                            'action_chosen': torch.zeros(args['batch_size'], 2, device=device), # left/right
                        }

                        _, hidden, w_hidden, hs = all_models[mdl_idx](all_x, steps=what_where_task.T_fixation, 
                                                        neumann_order=0,
                                                        hidden=hidden, w_hidden=w_hidden, 
                                                        DAs=None, save_all_states=True,
                                                        perturbation=current_perturbation if perturbation_phase == 1 else None)
                        results_dict['neuron_states'][-1][-1].append(hs)

                        ''' third phase, give stimuli and no feedback '''
                        all_x = {
                            'fixation': torch.ones(args['batch_size'], 1, device=device),
                            'stimulus': stim_inputs[i],  # Use perturbed stimulus inputs
                            'reward': torch.zeros(args['batch_size'], 2, device=device), # 2+2 for chosen and unchosen rewards
                            'action_chosen': torch.zeros(args['batch_size'], 2, device=device), # left/right
                        }

                        output, hidden, w_hidden, hs = all_models[mdl_idx](all_x, steps=what_where_task.T_stim, 
                                                            neumann_order=0,
                                                            hidden=hidden, w_hidden=w_hidden, 
                                                            DAs=None, save_all_states=True,
                                                            perturbation=current_perturbation if perturbation_phase == 2 else None)
                        results_dict['neuron_states'][-1][-1].append(hs)

                        ''' use output to calculate action, reward, and record loss function '''
                        action = torch.multinomial(output['action'].softmax(-1), num_samples=1).squeeze(-1) # (batch size, )
                        rwd_ch = rewards[i][torch.arange(args['batch_size']),action] # (batch size, )
                        
                        results_dict['img_chosen'][-1].append((action!=trial_info['stim_configs'][i])*1)
                        # (loc_choice, config)->img_chosen, (0,0)->0, (1,0)->1, (0,1)->1, (1,1)->0
                        results_dict['loc_chosen'][-1].append(action)
                        results_dict['reward'][-1].append(rwd_ch)

                        '''fourth phase, give stimuli and choice, and update weights'''
                        all_x = {
                            'fixation': torch.ones(args['batch_size'], 1, device=device),
                            'stimulus': stim_inputs[i],  # Use perturbed stimulus inputs here too
                            'reward': torch.eye(2, device=device)[None][torch.arange(args['batch_size']), rwd_ch], # reward/reward
                            'action_chosen': torch.eye(2, device=device)[None][torch.arange(args['batch_size']), action], # left/right
                        }

                        output, hidden, w_hidden, hs = all_models[mdl_idx](all_x, steps=what_where_task.T_choice_reward, 
                                                        neumann_order=0,
                                                        hidden=hidden, w_hidden=w_hidden, 
                                                        DAs=(2*rwd_ch-1), save_all_states=True,
                                                        perturbation=current_perturbation if perturbation_phase == 3 else None)
                        results_dict['neuron_states'][-1][-1].append(hs)
                        results_dict['synaptic_states'][-1].append(w_hidden)

                        results_dict['neuron_states'][-1][-1] = np.concatenate(results_dict['neuron_states'][-1][-1], axis=0) # num_timesteps, batch_size, num_dims

            # collect results in 
            results_dict['img_chosen'][-1] = np.array(results_dict['img_chosen'][-1]) # num_trials X batch_size
            results_dict['loc_chosen'][-1] = np.array(results_dict['loc_chosen'][-1]) # num_trials X batch_size 
            results_dict['reward'][-1] = np.array(results_dict['reward'][-1]) # num_trials X batch_size

            results_dict['neuron_states'][-1] = np.stack(results_dict['neuron_states'][-1]) # num_trials, num_timesteps, batch_size, num_dims
            results_dict['synaptic_states'][-1] = np.stack(results_dict['synaptic_states'][-1]) # num_trials X batch_size X num_dims X num_dims
        
    for k, v in results_dict.items():
        results_dict[k] = np.stack(v)

    for k, v in results_dict.items():
        print(k, v.shape)

    return results_dict


# Example usage:
if __name__ == "__main__":
    exp_dir = '/dartfs-hpc/rc/home/d/f005d7d/attn-rnn/what_where_analysis/what-where-task-analysis/rnn/exp'
    figure_data_dir = '/dartfs-hpc/rc/home/d/f005d7d/attn-rnn/what_where_analysis/what-where-task-analysis/figures'

    model_array_dir = [f'simple_setup_{i}' for i in range(1,9)]
    # model_array_dir = [f'test{i}' for i in range(1,9)]

    f = open(os.path.join(exp_dir, model_array_dir[0], 'args.json'), 'r')
    args = json.load(f)
    print('loaded args')

    what_where_task = WhatAndWhereTask(args['dt'], args['stim_dims'])

    input_config = {
        'fixation': (1, [0]),
        'stimulus': (args['stim_dims']*2, [0]),
        'reward': (2, [0]), 
        'action_chosen': (2, [0]), 
    }

    output_config = {
        'action': (2, [0]), # left, right
        'stimulus': (args['stim_dims'], [0]),
        'block_type': (2, [0]), # where or what block
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

    n_orthogonal_directions = 1
    aligned_perturbation_vectors, orthogonal_perturbation_vectors = \
        generate_perturbation_vector(all_models, n_orthogonal_directions=n_orthogonal_directions)
    
    test_activities_dir = '/dartfs/rc/lab/S/SoltaniA/f005d7d/what_where_analysis/rnn_test_activities/test_activities_with_perturbations.pkl'

    all_saved_states = {'aligned_perturbation': [], 'orthogonal_perturbation': []}

    print('testing perturbation aligned with block type readout')
    all_saved_states['aligned_perturbation'] = \
        test_model_with_input_perturbation(all_models, test_samples=1, perturbation_vectors=aligned_perturbation_vectors)

    print('testing perturbation orthogonal to block type readout')
    for i in range(n_orthogonal_directions):
        print(f'testing orthogonal perturbation {i} out of {n_orthogonal_directions}')
        all_saved_states['orthogonal_perturbation'].append(
            test_model_with_input_perturbation(all_models, test_samples=1, perturbation_vectors=orthogonal_perturbation_vectors[:, i, :]))

    print('simulation complete')
    with open(test_activities_dir, 'wb') as f:
        pickle.dump(all_saved_states, f)
    print(f'saved results to {test_activities_dir}')
