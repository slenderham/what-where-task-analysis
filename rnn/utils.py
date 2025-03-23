"""
Utilities
"""

from collections import Counter, OrderedDict
import json
import os
import pickle
import shutil

import numpy as np
import torch

random_counter = [0]

# TODO: add read list from file function
# TODO: add load model function

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, raw=False):
        self.raw = raw
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.raw:
            self.raw_scores = []

    def update(self, val, n=1, raw_scores=None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.raw:
            self.raw_scores.extend(list(raw_scores))


def save_checkpoint(state, is_best, folder='./',
                    filename='checkpoint.pth.tar', 
                    best_filename='model_best.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, best_filename))

def load_checkpoint(model, optimizer, device, folder='./', filename='checkpoint.pth.tar'):
    checkpoint = torch.load(os.path.join(folder, filename), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def merge_args_with_dict(args, dic):
    for k, v in list(dic.items()):
        setattr(args, k, v)

def make_output_and_sample_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sample_dir = os.path.join(out_dir, 'samples')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    return out_dir, sample_dir

def save_defaultdict_to_fs(d, out_path):
    d = dict(d)
    with open(out_path, 'w') as fp:
        d_str = json.dumps(d, ensure_ascii=True)
        fp.write(d_str)

def save_list_to_fs(l, out_path):
    with open(out_path, 'w') as fp:
        for n in l:
            fp.write(str(n)+'\n')

def load_list_from_fs(outpath):
    with open(outpath, 'r') as fp:
        l = fp.read().split('\n')[:-1]
        return [float(n) for n in l]

def add_weight_decay_group(model, weight_decay):
    decay = dict()
    no_decay = dict()
    for n, p in model.named_parameters():
        if 'weight' in n:
            decay.append(n)
        else:
            no_decay.append(p)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def get_area_ei_indices(args, ind_area):
    hidden_size_per_area = args['hidden_size']
    E_SIZE = round(args['hidden_size']*args['e_prop'])
    I_SIZE = round(args['hidden_size']*(1-args['e_prop']))
    NUM_AREAS = args['num_areas']
    return np.concatenate([np.arange(E_SIZE*ind_area, E_SIZE*(ind_area+1)), 
                           np.arange(E_SIZE*NUM_AREAS+I_SIZE*ind_area, E_SIZE*NUM_AREAS+I_SIZE*(ind_area+1))])