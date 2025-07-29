import numpy as np
import torch
from matplotlib import pyplot as plt
import os
import torch.nn.functional as F
from scipy.stats import norm

class WhatAndWhereTask():
    def __init__(self, dt, stim_dims):

        self.times = {
            'ITI': 0.4,
            'fixation_time': 0.4,
            'stim_time': 0.8,
            'choice_reward_time': 0.6
        }
        self.dt = dt

        self.T_ITI = int(self.times['ITI']/self.dt)
        self.T_fixation = int(self.times['fixation_time']/self.dt)
        self.T_stim = int(self.times['stim_time']/self.dt)
        self.T_choice = int(self.times['choice_reward_time']/self.dt)

        self.stim_dims = stim_dims

    def generate_trials(self, batch_size, trials_per_block, reversal_interval, reward_schedule=None,
                        stim_orders=None, block_type=None):
        
        if reward_schedule is None:
            train_reward_prob_high = np.random.rand()*0.3+0.6 # uniform [0.6, 0.9]
            reward_schedule = [train_reward_prob_high, 1-train_reward_prob_high]
        
        all_trials_info = {k: [] for k in [
            'stim_configs',
            'stim_inputs',
            'rewards',
            'reward_probs',
            'targets'
        ]}
        for sess_idx in range(batch_size):
            curr_sess_trials_info = self._generate_single_trial(trials_per_block, reversal_interval, reward_schedule,
                                                                stim_orders, block_type)
            for k in all_trials_info.keys():
                all_trials_info[k].append(curr_sess_trials_info[k])

        for k in all_trials_info.keys():
            all_trials_info[k] = torch.from_numpy(np.stack(all_trials_info[k], axis=1))

        return all_trials_info
    
    def _generate_single_trial(self, trials_per_block, reversal_interval, reward_schedule,
                               stim_orders=None, block_type=None):
        '''sample block type if none is provided'''
        if block_type is None:
            block_type = np.random.randint(2)            

        '''sample stim configurations'''
        # (n_trials, )
        # the same as stim on the left
        if stim_orders is None:
            stim_configs = np.random.permutation(np.repeat(
                np.array([0, 1]), repeats=trials_per_block//2
            ))
        else:
            stim_configs = stim_orders

        '''sample stim input to network'''
        img_reps = np.random.randn(self.stim_dims)
        img_reps = img_reps/np.linalg.norm(img_reps)
        img_reps = np.stack([1+img_reps, 1-img_reps], axis=0)
        stim_inputs = np.concatenate([
            img_reps[stim_configs], img_reps[1-stim_configs]
        ], axis=-1) # (n_trials, 2*n_dims)
        
        '''get reward probabilities'''
        initial_better_option = np.random.randint(2)
        reversal_point = np.random.randint(reversal_interval[0]-1, reversal_interval[1])
        reward_probs = np.empty((trials_per_block, 2))*np.nan
        if block_type==0:
            # for where blocks, the reward probabilities are tied to the location
            reward_probs[:reversal_point, initial_better_option] = reward_schedule[0]
            reward_probs[:reversal_point, 1-initial_better_option] = reward_schedule[1]
            reward_probs[reversal_point:, initial_better_option] = reward_schedule[1]
            reward_probs[reversal_point:, 1-initial_better_option] = reward_schedule[0]
        elif block_type==1:
            # for what blocks, the reward probabilities are tied to the stimulus
            for trial_idx in range(trials_per_block):
                if trial_idx<reversal_point:
                    if stim_configs[trial_idx]==initial_better_option:
                        reward_probs[trial_idx, 0] = reward_schedule[0] 
                        reward_probs[trial_idx, 1] = reward_schedule[1]
                    else:
                        reward_probs[trial_idx, 0] = reward_schedule[1] 
                        reward_probs[trial_idx, 1] = reward_schedule[0]
                else:
                    if stim_configs[trial_idx]==initial_better_option:
                        reward_probs[trial_idx, 0] = reward_schedule[1] 
                        reward_probs[trial_idx, 1] = reward_schedule[0]
                    else:
                        reward_probs[trial_idx, 0] = reward_schedule[0] 
                        reward_probs[trial_idx, 1] = reward_schedule[1]                    
        else:
            raise ValueError
        
        '''sample rewards'''
        rewards = np.empty_like(reward_probs)*np.nan

        # pre-reversal rewards
        pre_rev_num_rewards_high = int(np.round(reward_schedule[0]*reversal_point))
        rewards[:reversal_point][np.isclose(reward_probs[:reversal_point], reward_schedule[0])] = \
            np.random.permutation(np.concatenate([
                np.ones(pre_rev_num_rewards_high), np.zeros(reversal_point-pre_rev_num_rewards_high)
            ]))
        pre_rev_num_rewards_low = int(np.round(reward_schedule[1]*reversal_point))
        rewards[:reversal_point][np.isclose(reward_probs[:reversal_point], reward_schedule[1])] = \
            np.random.permutation(np.concatenate([
                np.ones(pre_rev_num_rewards_low), np.zeros(reversal_point-pre_rev_num_rewards_low)
            ]))
        
        # post-reversal rewards
        post_rev_num_rewards_high = int(np.round(reward_schedule[0]*(trials_per_block-reversal_point)))
        rewards[reversal_point:][np.isclose(reward_probs[reversal_point:], reward_schedule[0])] = \
            np.random.permutation(np.concatenate([
                np.ones(post_rev_num_rewards_high), np.zeros(trials_per_block-reversal_point-post_rev_num_rewards_high)
            ]))
        post_rev_num_rewards_low = int(np.round(reward_schedule[1]*(trials_per_block-reversal_point)))
        rewards[reversal_point:][np.isclose(reward_probs[reversal_point:], reward_schedule[1])] = \
            np.random.permutation(np.concatenate([
                np.ones(post_rev_num_rewards_low), np.zeros(trials_per_block-reversal_point-post_rev_num_rewards_low)
            ]))
        
        return {
            'stim_configs': stim_configs,
            'stim_inputs': stim_inputs,
            'rewards': rewards.astype(int),
            'reward_probs': reward_probs,
            'targets': np.argmax(reward_probs, -1)
        }

if __name__=='__main__':
    task = WhatAndWhereTask(0.02, 2)

    all_trials_info = task.generate_trials(1, 40, [15, 25])

    for k,v in all_trials_info.items():
        print(k, v.shape)

    print(np.unique(all_trials_info['stim_configs'], return_counts=True))
    print(np.unique(all_trials_info['rewards'], return_counts=True))
    print(np.unique(all_trials_info['reward_probs'], return_counts=True))
    print(np.unique(all_trials_info['targets'], return_counts=True))