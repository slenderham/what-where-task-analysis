import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import time

from models_optimized import HierarchicalPlasticRNN
from task import WhatAndWhereTask
from utils import (AverageMeter, load_checkpoint, load_list_from_fs,
                   save_checkpoint, save_defaultdict_to_fs, save_list_to_fs)
import wandb

def train(model, iters):
    model.train()
    optimizer.zero_grad()
    pbar = tqdm(total=iters)
    # to save the accuracy of the action and block type in preparation and choice
    total_acc = {'action_acc': 0, 'block_type_acc': 0}
    # to save the loss during fixation period: fixation, preparatory block type, 
    # and loss during choice period: action, stimulus, block type, and saccade
    total_loss = {'dv': 0, 'action': 0, 'stimulus': 0, 'block_type': 0}

    num_loss_components = 4
    num_action_weight = 1/(num_loss_components*1) # fixation for fixation phase, action for the choice phase
    num_stimulus_weight = 1/(num_loss_components*1) # only for the stimulus phase
    num_block_type_weight = 1/(num_loss_components*1) # for fixation and stimulus phase
    num_dv_weight = 1/(num_loss_components*1) # for the stimulus phase

    uniform_probs = torch.ones(args.batch_size, 2, device=device)/2 # (batch_size, 2), used for forcing equal probability

    p_delay = 0.5
    delay_mask = torch.randperm(iters)<(iters*p_delay) # mask for whether to add a delay phase

    for batch_idx in range(iters):
        trial_info = what_where_task.generate_trials(
            batch_size = args.batch_size,
            trials_per_block = args.trials_per_block, 
            reversal_interval = [args.trials_per_block//2-args.reversal_interval_range//2, 
                                 args.trials_per_block//2+args.reversal_interval_range//2,],
        ) 
        stim_inputs = trial_info['stim_inputs'].to(device, dtype=torch.float) 
        rewards = trial_info['rewards'].to(device)
        action_targets = trial_info['action_targets'].to(device)
        block_type_target = trial_info['block_types'].to(device) # (batch_size, )
        stimulus_targets = trial_info['stimulus_targets'].to(device) # (batch_size, 2)
        stimulus_configs = trial_info['stim_configs'].to(device) # (batch_size, )
        
        loss = 0
        hidden, w_hidden = model.init_hidden(args.batch_size, device)
        
        # the direction of update, based on the action and stimulus information
        # set this to be None for the first trial
        num_phases = 5 if delay_mask[batch_idx] else 4

        for i in range(len(stim_inputs)):
            ''' first phase, give nothing '''
            all_x = {
                'go_cue': torch.zeros(args.batch_size, 1, device=device),
                'fixation': torch.zeros(args.batch_size, 1, device=device),
                'stimulus': torch.zeros_like(stim_inputs[i]),
                'reward': torch.zeros(args.batch_size, 2, device=device), # no reward/reward
                'action_chosen': torch.zeros(args.batch_size, 2, device=device), # left/right
            }
            output, hidden, w_hidden, hs = model(all_x, steps=what_where_task.T_ITI, 
                                            neumann_order=args.neumann_order,
                                            hidden=hidden, w_hidden=w_hidden, 
                                            DAs=torch.zeros(args.batch_size, device=device))
            # regularize firing rate
            loss += args.l2r*hs.pow(2).mean()/num_phases

            ''' second phase, fixation at middle '''
            all_x = {
                'go_cue': torch.zeros(args.batch_size, 1, device=device),
                'fixation': torch.ones(args.batch_size, 1, device=device),
                'stimulus': torch.zeros_like(stim_inputs[i]),
                'reward': torch.zeros(args.batch_size, 2, device=device), # no reward/reward
                'action_chosen': torch.zeros(args.batch_size, 2, device=device), # left/right
            }
            output, hidden, w_hidden, hs = model(all_x, steps=what_where_task.T_fixation, 
                                            neumann_order=args.neumann_order,
                                            hidden=hidden, w_hidden=w_hidden, 
                                            DAs=torch.zeros(args.batch_size, device=device))
            
            # decode action to be fixation
            output_action = output['action'].flatten(end_dim=-2) # (batch_size, 2)
            loss += F.kl_div(F.log_softmax(output_action, dim=-1), uniform_probs, reduction='batchmean')*num_action_weight
            total_loss['action'] += F.kl_div(F.log_softmax(output_action.detach(), dim=-1), uniform_probs, reduction='batchmean').detach().item()/len(stim_inputs)

            # decode block type 
            output_block_type = output['block_type'].flatten(end_dim=-2) # (batch_size, 2)
            target_block_type = block_type_target[i].flatten() # (batch_size)
            loss += F.cross_entropy(output_block_type, target_block_type, 
                                    weight=0.5/torch.tensor(what_where_task.block_type_probs))*num_block_type_weight
            total_loss['block_type'] += F.cross_entropy(output_block_type.detach(), target_block_type, 
                                                        weight=0.5/torch.tensor(what_where_task.block_type_probs)).detach().item()/len(stim_inputs)
            total_acc['block_type_acc'] += (output_block_type.argmax(dim=-1)==target_block_type).float().item()/len(stim_inputs)/2

            # regularize firing rate
            loss += args.l2r*hs.pow(2).mean()/num_phases

            ''' optional third phase, give stimuli and no feedback, no go cue and only readout values'''
            if delay_mask[batch_idx]:
                all_x = {
                    'go_cue': torch.zeros(args.batch_size, 1, device=device),
                    'fixation': torch.ones(args.batch_size, 1, device=device),
                    'stimulus': stim_inputs[i],
                    'reward': torch.zeros(args.batch_size, 2, device=device), # no reward/reward
                    'action_chosen': torch.zeros(args.batch_size, 2, device=device), # left/right
                }
                output, hidden, w_hidden, hs = model(all_x, steps=what_where_task.T_delay, 
                                                neumann_order=args.neumann_order,
                                                hidden=hidden, w_hidden=w_hidden, 
                                                DAs=torch.zeros(args.batch_size, device=device))

                # decode action to be fixation
                output_action = output['action'].flatten(end_dim=-2) # (batch_size, 2)
                loss += F.kl_div(F.log_softmax(output_action, dim=-1), uniform_probs, reduction='batchmean')*num_action_weight/p_delay
                total_loss['action'] += F.kl_div(F.log_softmax(output_action.detach(), dim=-1), uniform_probs, reduction='batchmean').detach().item()/len(stim_inputs)

                # decode loc_update and img_update directions
                output_dv_loc = output['dv_loc'].flatten(end_dim=-2) # (batch_size, 2)
                output_dv_stim = output['dv_stim'].flatten(end_dim=-2) # (batch_size, 2)
                target_action = action_targets[i].flatten() # (batch_size, )
                if block_type_target[i] == 0:
                    loss += F.cross_entropy(output_dv_loc, target_action)*num_dv_weight/p_delay
                    loss += F.kl_div(F.log_softmax(output_dv_stim, dim=-1), uniform_probs, reduction='batchmean')*num_dv_weight/p_delay
                    total_loss['dv'] += F.cross_entropy(output_dv_loc.detach(), target_action).detach().item()/len(stim_inputs)/p_delay
                    total_loss['dv'] += F.kl_div(F.log_softmax(output_dv_stim.detach(), dim=-1), uniform_probs, reduction='batchmean').detach().item()/len(stim_inputs)/p_delay
                else:
                    loss += F.kl_div(F.log_softmax(output_dv_loc, dim=-1), uniform_probs, reduction='batchmean')*num_dv_weight/p_delay
                    loss += F.cross_entropy(output_dv_stim, target_action)*num_dv_weight/p_delay
                    total_loss['dv'] += F.kl_div(F.log_softmax(output_dv_loc.detach(), dim=-1), uniform_probs, reduction='batchmean').detach().item()/len(stim_inputs)/p_delay
                    total_loss['dv'] += F.cross_entropy(output_dv_stim.detach(), target_action).detach().item()/len(stim_inputs)/p_delay
            
                # decode block type
                output_block_type = output['block_type'].flatten(end_dim=-2) # (batch_size, 2)
                target_block_type = block_type_target[i].flatten() # (batch_size)
                loss += F.cross_entropy(output_block_type, target_block_type, 
                                        weight=0.5/torch.tensor(what_where_task.block_type_probs))*num_block_type_weight/p_delay
                total_loss['block_type'] += F.cross_entropy(output_block_type.detach(), target_block_type, 
                                                            weight=0.5/torch.tensor(what_where_task.block_type_probs)).detach().item()/len(stim_inputs)/p_delay
                total_acc['block_type_acc'] += (output_block_type.argmax(dim=-1)==target_block_type).float().item()/len(stim_inputs)/p_delay/2

                # regularize firing rate
                loss += args.l2r*hs.pow(2).mean()/num_phases
            

            ''' fourth phase, give stimuli and go cue, read out action'''
            all_x = {
                'go_cue': torch.ones(args.batch_size, 1, device=device),
                'fixation': torch.ones(args.batch_size, 1, device=device),
                'stimulus': stim_inputs[i],
                'reward': torch.zeros(args.batch_size, 2, device=device), # no reward/reward
                'action_chosen': torch.zeros(args.batch_size, 2, device=device), # left/right
            }
            output, hidden, w_hidden, hs = model(all_x, steps=what_where_task.T_stim, 
                                                neumann_order=args.neumann_order,
                                                hidden=hidden, w_hidden=w_hidden, 
                                                DAs=torch.zeros(args.batch_size, device=device))

            #  use output to calculate action, reward, and record loss function
            action = torch.multinomial(output['action'].softmax(-1), num_samples=1).squeeze(-1) # (batch size, )
            rwd_ch = rewards[i][torch.arange(args.batch_size),action] # (batch size, )
           
            # decode action
            output_action = output['action'].flatten(end_dim=-2) # (batch_size, 2)
            target_action = action_targets[i].flatten() # (batch_size, )
            loss += F.cross_entropy(output_action, target_action)*num_action_weight
            total_loss['action'] += F.cross_entropy(output_action.detach(), target_action).detach().item()/len(stim_inputs)
            total_acc['action_acc'] += (action==target_action).float().item()/len(stim_inputs)

            # decode stimulus
            output_stimulus = output['stimulus'] # (batch_size, 4)
            output_stimulus = output_stimulus[...,[1,3]]-output_stimulus[...,[0,2]] # (batch_size, 2)
            output_stimulus = output_stimulus.flatten() # (batch_size, 2)
            target_stimulus = stimulus_targets[i].flatten() # (batch_size, 2)
            loss += F.mse_loss(output_stimulus, target_stimulus)*num_stimulus_weight
            total_loss['stimulus'] += F.mse_loss(output_stimulus.detach(), target_stimulus).detach().item()/len(stim_inputs)

            # regularize firing rates
            loss += args.l2r*hs.pow(2).mean()/num_phases  # + args.l1r*hs.abs().mean()
            
            '''fifth phase, give stimuli and choice, and update weights'''
            all_x = {
                'go_cue': torch.ones(args.batch_size, 1, device=device),
                'fixation': torch.ones(args.batch_size, 1, device=device),
                'stimulus': stim_inputs[i],  # only chosen stimulus input, zero the other one
                'reward': torch.eye(2, device=device)[None][torch.arange(args.batch_size), rwd_ch], # no reward/reward
                'action_chosen': torch.eye(2, device=device)[None][torch.arange(args.batch_size), action], # left/right
            }
            output, hidden, w_hidden, hs = model(all_x, steps=what_where_task.T_choice_reward, 
                                            neumann_order=args.neumann_order,
                                            hidden=hidden, w_hidden=w_hidden, 
                                            DAs=(2*rwd_ch-1).float())

            # regularize firing rate
            loss += args.l2r*hs.pow(2).mean()/num_phases # + args.l1r*hs.abs().mean()

            # regularize weight
            loss += args.l2w*w_hidden.pow(2).sum(dim=(-2, -1)).mean()
            if args.num_areas>1:
                loss += args.l1w*(model.mask_rec_inter*w_hidden).abs().sum(dim=(-2,-1)).mean()

        loss /= len(stim_inputs)
        
        # add weight decay for static weights
        for input_w in model.rnn.x2h.values():
            loss += args.l2w*(input_w.effective_weight().pow(2).sum())
        for output_w in model.h2o.values():
            loss += args.l2w*(output_w.effective_weight().pow(2).sum())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
        optimizer.step()
        optimizer.zero_grad()
        # clamp weight values to 
        with torch.no_grad():
            model.rnn.h2h.weight.data.clamp_(-model.plasticity.weight_bound, model.plasticity.weight_bound)
            
        if (batch_idx+1) % log_interval == 0:
            if torch.isnan(loss):
                print('Overflown loss')
                quit()
            pbar.set_description('Iteration {} Loss: {:.3f}, {:.3f}, {:.3f}, {:.3f}; Acc: {:.3f}, {:.3f}'.format(
                batch_idx+1, 
                total_loss['action']/(batch_idx+1), total_loss['dv']/(batch_idx+1),
                total_loss['stimulus']/(batch_idx+1), total_loss['block_type']/(batch_idx+1), 
                total_acc['action_acc']/(batch_idx+1), total_acc['block_type_acc']/(batch_idx+1)))
            # pbar.refresh()
            pbar.update(log_interval)
    pbar.close()
    for k, v in total_acc.items():
        total_acc[k] = total_acc[k]/iters
    for k, v in total_loss.items():
        total_loss[k] = total_loss[k]/iters
    print(f'Training Loss: {[f"{k}: {v:.4f}" for k, v in total_loss.items()]}; Training Acc: {[f"{k}: {v:.4f}" for k, v in total_acc.items()]}')
    wandb.log({**total_loss, **total_acc})
    return loss.item()


def eval(model, epoch):
    model.eval()
    with torch.no_grad():
        losses_means_by_block_type = []
        losses_stds_by_block_type = []
        for test_block_type in range(2):
            curr_block_losses = []
            for batch_idx in range(args.eval_samples):
                trial_info = what_where_task.generate_trials(
                    batch_size = args.batch_size,
                    trials_per_block = args.trials_per_test_block, 
                    reversal_interval = [args.trials_per_test_block//2-args.test_reversal_interval_range//2, 
                                        args.trials_per_test_block//2+args.test_reversal_interval_range//2,],
                    reward_schedule=[args.reward_probs_high, 1-args.reward_probs_high],
                    block_type=test_block_type,
                ) 
                stim_inputs = trial_info['stim_inputs'].to(device, dtype=torch.float)
                rewards = trial_info['rewards'].to(device)
                targets = trial_info['action_targets'].to(device)
                
                loss = []
                hidden, w_hidden = model.init_hidden(args.batch_size, device)
                
                for i in range(len(stim_inputs)):
                    ''' first phase, give nothing '''
                    all_x = {
                        'go_cue': torch.zeros(args.batch_size, 1, device=device),
                        'fixation': torch.zeros(args.batch_size, 1, device=device),
                        'stimulus': torch.zeros_like(stim_inputs[i]),
                        'reward': torch.zeros(args.batch_size, 2, device=device), # no reward/reward
                        'action_chosen': torch.zeros(args.batch_size, 2, device=device), # left/right
                    }

                    _, hidden, w_hidden, hs = model(all_x, steps=what_where_task.T_ITI, 
                                                    neumann_order=args.neumann_order,
                                                    hidden=hidden, w_hidden=w_hidden, 
                                                    DAs=torch.zeros(args.batch_size, device=device))
                    
                    ''' second phase, give fixation '''
                    all_x = {
                        'go_cue': torch.zeros(args.batch_size, 1, device=device),
                        'fixation': torch.ones(args.batch_size, 1, device=device),
                        'stimulus': torch.zeros_like(stim_inputs[i]),
                        'reward': torch.zeros(args.batch_size, 2, device=device), # no reward/reward
                        'action_chosen': torch.zeros(args.batch_size, 2, device=device), # left/right
                    }

                    _, hidden, w_hidden, hs = model(all_x, steps=what_where_task.T_fixation, 
                                                    neumann_order=args.neumann_order,
                                                    hidden=hidden, w_hidden=w_hidden, 
                                                    DAs=torch.zeros(args.batch_size, device=device))

                    ''' third phase, give stimuli and no feedback. always give go cue at stimulus onset'''
                    all_x = {
                        'go_cue': torch.ones(args.batch_size, 1, device=device),
                        'fixation': torch.ones(args.batch_size, 1, device=device),
                        'stimulus': stim_inputs[i],
                        'reward': torch.zeros(args.batch_size, 2, device=device), # no reward/reward
                        'action_chosen': torch.zeros(args.batch_size, 2, device=device), # left/right
                    }

                    output, hidden, w_hidden, hs = model(all_x, steps=what_where_task.T_stim, 
                                                        neumann_order=args.neumann_order,
                                                        hidden=hidden, w_hidden=w_hidden, 
                                                        DAs=torch.zeros(args.batch_size, device=device))

                    ''' use output to calculate action, reward, and record loss function '''
                    action = torch.multinomial(output['action'].softmax(-1), num_samples=1).squeeze(-1) # (batch size, )
                    rwd_ch = rewards[i][range(args.batch_size),action] # (batch size, )
                    loss.append((action==targets[i]).float()) # (batch size, )

                    '''fourth phase, give stimuli and choice, and update weights'''
                    all_x = {
                        'go_cue': torch.ones(args.batch_size, 1, device=device),
                        'fixation': torch.ones(args.batch_size, 1, device=device),
                        'stimulus': stim_inputs[i],
                        'reward': torch.eye(2, device=device)[None][torch.arange(args.batch_size), rwd_ch], # reward/reward
                        'action_chosen': torch.eye(2, device=device)[None][torch.arange(args.batch_size), action], # left/right
                    }

                    output, hidden, w_hidden, hs = model(all_x, steps=what_where_task.T_choice_reward, 
                                                    neumann_order=args.neumann_order,
                                                    hidden=hidden, w_hidden=w_hidden, 
                                                    DAs=(2*rwd_ch-1).float())
        
                loss = torch.stack(loss, dim=0) # shape: (trials_per_test_block, batch_size)
                curr_block_losses.append(loss) # (eval_samples, trials_per_test_block, batch_size)
            # Save all losses for this block type
            losses_means_by_block_type.append(torch.stack(curr_block_losses, dim=0).mean(dim=(0,2))) # (trials_per_test_block)
            losses_stds_by_block_type.append(torch.stack(curr_block_losses, dim=0).std(dim=(0,2))) # (trials_per_test_block)
        print('====> Epoch {} Eval Loss Where Block: {:.3f}, What Block: {:.3f}'.format(
            epoch+1, losses_means_by_block_type[0].mean(), losses_means_by_block_type[1].mean()))
        wandb.log({
            'Eval loss where block': losses_means_by_block_type[0].mean(),
            'Eval loss what block': losses_means_by_block_type[1].mean()
        })
        return torch.stack(losses_means_by_block_type, dim=0), torch.stack(losses_stds_by_block_type, dim=0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='Output directory')
    parser.add_argument('--iters', type=int, help='Training iterations')
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--stim_dims', type=int, default=2, help='Size of stimulus input')
    parser.add_argument('--hidden_size', type=int, default=80, help='Size of recurrent layer')
    parser.add_argument('--num_areas', type=int, default=1, help='Number of recurrent areas')
    parser.add_argument('--trials_per_block', type=int, default=60, help='Number of trials')
    parser.add_argument('--trials_per_test_block', type=int, default=80, help='Number of trials')
    parser.add_argument('--reward_probs_high', type=float, default=0.7, help='Reward probability of better option')
    parser.add_argument('--reversal_interval_range', type=int, default=40, help='Range of reversal interval')
    parser.add_argument('--test_reversal_interval_range', type=int, default=20, help='Range of reversal interval')
    parser.add_argument('--e_prop', type=float, default=4/5, help='Proportion of E neurons')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--neumann_order', type=int, default=10, help='Timestep for unrolling for neumann approximation')
    parser.add_argument('--eval_samples', type=int, default=30, help='Number of samples to use for evaluation.')
    parser.add_argument('--max_norm', type=float, default=1.0, help='Max norm for gradient clipping')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sigma_in', type=float, default=0.01, help='Std for input noise')
    parser.add_argument('--sigma_rec', type=float, default=0.1, help='Std for recurrent noise')
    parser.add_argument('--sigma_w', type=float, default=0.01, help='Std for weight noise')
    parser.add_argument('--init_spectral', type=float, default=1.0, help='Initial spectral radius for the recurrent weights')
    parser.add_argument('--balance_ei', action='store_true', help='Make mean of E and I recurrent weights equal')
    parser.add_argument('--tau_x', type=float, default=0.1, help='Time constant for recurrent neurons')
    parser.add_argument('--tau_w', type=float, default=200, help='Time constant for weight modification')
    parser.add_argument('--dt', type=float, default=0.02, help='Discretization time step (ms)')
    parser.add_argument('--l2r', type=float, default=0.0, help='Weight for L2 reg on firing rate')
    parser.add_argument('--l2w', type=float, default=0.0, help='Weight for L2 reg on weight')
    parser.add_argument('--l1r', type=float, default=0.0, help='Weight for L1 reg on firing rate')
    parser.add_argument('--l1w', type=float, default=0.0, help='Weight for L1 reg on weight')
    parser.add_argument('--plas_type', type=str, choices=['all', 'none'], default='all', help='How much plasticity')
    parser.add_argument('--activ_func', type=str, choices=['relu', 'softplus', 'softplus2', 'retanh', 'sigmoid'], 
                        default='retanh', help='Activation function for recurrent units')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--save_checkpoint', action='store_true', help='Whether to save the trained model')
    parser.add_argument('--load_checkpoint', action='store_true', help='Whether to load the trained model')
    parser.add_argument('--debug', action='store_true', help='Debug mode for plotting')
    parser.add_argument('--cuda', action='store_true', help='Enables CUDA training')

    args = parser.parse_args()

    if args.save_checkpoint:
        print(f"Parameters saved to {os.path.join(args.exp_dir, 'args.json')}")
        save_defaultdict_to_fs(vars(args), os.path.join(args.exp_dir, 'args.json'))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if (not torch.cuda.is_available()):
        print("No CUDA available so not using it")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if args.cuda else 'cpu')

    log_interval = 100
    what_where_task = WhatAndWhereTask(args.dt, args.stim_dims)

    input_config = {
        'fixation': (1, [0]), # fixation input, 0 or 1
        'go_cue': (1, [0]), # go cue input, 0 or 1
        'stimulus': (args.stim_dims*2, [0]), # stimulus input, concatenated left and right stimulus
        'reward': (2, [0]), # reward input, [0 1] or [1 0]
        'action_chosen': (2, [0]), # action chosen input, left or right
    }

    # decode the action and stimulus to choose at the end of fixation and during choice using separate readouts
    # also decode the block type
    output_config = {
        'action': (2, [0], True), # action value decoding, left, right, 
        'stimulus': (args.stim_dims*2, [0], True), # stimulus value decoding,stimulus
        'block_type': (2, [0], True), # block type decoding, where or what block
        # decode the desired direction for next choice, separately for block types
        # detach gradient so that they don't affect the rest of the model
        'dv_loc': (2, [0], True), # desired location based on previous trial outcome
        'dv_stim': (2, [0], True), # location of desired stimulus based on previous trial outcome
    }

    total_trial_time = what_where_task.times['ITI']+\
                        what_where_task.times['fixation_time']+\
                        what_where_task.times['stim_time']+\
                        what_where_task.times['choice_reward_time']

    model_specs = {'input_config': input_config, 'hidden_size': args.hidden_size, 'output_config': output_config,
                   'num_areas': args.num_areas, 'plastic': args.plas_type=='all', 'activation': args.activ_func,
                   'dt_x': args.dt, 'dt_w': total_trial_time, 'tau_x': args.tau_x, 'tau_w': args.tau_w, 
                   'e_prop': args.e_prop, 'init_spectral': args.init_spectral, 'balance_ei': args.balance_ei,
                   'sigma_rec': args.sigma_rec, 'sigma_in': args.sigma_in, 'sigma_w': args.sigma_w, 
                   'inter_regional_sparsity': (1, 1), 'inter_regional_gain': (1, 1)}
    
    model = HierarchicalPlasticRNN(**model_specs).to(device)
    model.compile(dynamic=False, fullgraph=True)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    print(model)
    for n, p in model.named_parameters():
        print(n, p.numel())
    print(optimizer)

    if args.load_checkpoint:
        load_checkpoint(model, optimizer, device, folder=args.exp_dir, filename='checkpoint.pth.tar')
        print('Model loaded successfully')

    wandb.init(project="what-where-rnn", config=args)

    metrics = defaultdict(list)
    best_eval_loss = 0
    for i in range(args.epochs):
        training_loss = train(model, args.iters)
        eval_loss_means, eval_loss_stds = eval(model, i)
        # lr_scheduler.step()
        metrics['eval_losses_mean'].append(eval_loss_means.squeeze().tolist())
        metrics['eval_losses_std'].append(eval_loss_stds.squeeze().tolist())
        metrics = dict(metrics)
        save_defaultdict_to_fs(metrics, os.path.join(args.exp_dir, 'metrics.json'))
        if args.save_checkpoint:
            if (eval_loss_means).mean().item() > best_eval_loss:
                is_best_epoch = True
                best_eval_loss = eval_loss_means.mean().item()
                metrics['best_epoch'] = i
                metrics['best_eval_loss'] = best_eval_loss
            else:
                is_best_epoch = False
            save_checkpoint({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                            is_best=is_best_epoch, folder=args.exp_dir, filename='checkpoint.pth.tar', 
                            best_filename='checkpoint_best.pth.tar')
    
    print('====> DONE')