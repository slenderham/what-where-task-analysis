import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

@torch.jit.script
def retanh(x):
    return torch.tanh(F.relu(x))

def _get_activation_function(func_name):
    if func_name=='relu':
        return F.relu
    elif func_name=='softplus':
        return lambda x: F.softplus(x-1)
    elif func_name=='softplus2':
        return lambda x: x/(1-torch.exp(-x))
    elif func_name=='retanh':
        return retanh
    elif func_name=='sigmoid':
        return lambda x: torch.tanh(x-1)+1
    else:
        raise RuntimeError(F"{func_name} is an invalid activation function.")

def _get_pos_function(func_name):
    if func_name=='relu':
        return F.relu
    if func_name=='abs':
        return torch.abs
    else:
        raise RuntimeError(F"{func_name} is an invalid function enforcing positive weight.")

def _get_plasticity_mask_rec(rec_mask, num_areas, e_hidden_units_per_area, i_hidden_units_per_area):
    plas_mask = {}
    plas_mask_ee = torch.kron(rec_mask, torch.ones(e_hidden_units_per_area, e_hidden_units_per_area))
    plas_mask_ie = torch.kron(rec_mask, torch.ones(i_hidden_units_per_area, e_hidden_units_per_area))
    plas_mask_ei = torch.zeros(num_areas*e_hidden_units_per_area, num_areas*i_hidden_units_per_area)
    plas_mask_ii = torch.zeros(num_areas*i_hidden_units_per_area, num_areas*i_hidden_units_per_area)
    plas_mask = torch.cat([
                    torch.cat([plas_mask_ee, plas_mask_ei], dim=1),
                    torch.cat([plas_mask_ie, plas_mask_ii], dim=1)], dim=0)
    plas_mask *= 1-torch.eye(plas_mask.shape[0])
    return plas_mask

def _get_connectivity_mask_in(in_mask, input_units, e_hidden_units_per_area, i_hidden_units_per_area):
    in_mask_e = torch.kron(in_mask, torch.ones(e_hidden_units_per_area, input_units))
    in_mask_i = torch.kron(in_mask, torch.ones(i_hidden_units_per_area, input_units))
    conn_mask = torch.cat([in_mask_e, in_mask_i], dim=0)
    return conn_mask

def _get_connectivity_mask_out(out_mask, output_units, e_hidden_units_per_area, i_hidden_units_per_area):
    out_mask_e = torch.kron(out_mask, torch.ones(output_units, e_hidden_units_per_area))
    out_mask_i = torch.kron(out_mask, torch.zeros(output_units, i_hidden_units_per_area))
    conn_mask = torch.cat([out_mask_e, out_mask_i], dim=1)
    return conn_mask

def _get_connectivity_mask_rec(rec_mask, num_areas, e_hidden_units_per_area, i_hidden_units_per_area):
    conn_mask = {}
    # recurrent connection mask
    rec_mask_ee = torch.kron(rec_mask, torch.ones(e_hidden_units_per_area, e_hidden_units_per_area))
    rec_mask_ie = torch.kron(rec_mask, torch.ones(i_hidden_units_per_area, e_hidden_units_per_area))
    rec_mask_ei = torch.kron(torch.eye(num_areas), torch.ones(e_hidden_units_per_area, i_hidden_units_per_area))
    rec_mask_ii = torch.kron(torch.eye(num_areas), torch.ones(i_hidden_units_per_area, i_hidden_units_per_area))
    conn_mask['rec'] = torch.cat([
                    torch.cat([rec_mask_ee, rec_mask_ei], dim=1),
                    torch.cat([rec_mask_ie, rec_mask_ii], dim=1)], dim=0)

    # within region and cross_region connectivity
    rec_mask_ee_intra = torch.kron(torch.eye(num_areas), torch.ones(e_hidden_units_per_area, e_hidden_units_per_area))
    rec_mask_ie_intra = torch.kron(torch.eye(num_areas), torch.ones(i_hidden_units_per_area, e_hidden_units_per_area))
    conn_mask['rec_intra'] = torch.cat([
                    torch.cat([rec_mask_ee_intra, rec_mask_ei], dim=1),
                    torch.cat([rec_mask_ie_intra, rec_mask_ii], dim=1)], dim=0)

    # feedforward connectivity
    rec_mask_ee_intra_ff = torch.kron(torch.diag(torch.ones(num_areas-1),-1), torch.ones(e_hidden_units_per_area, e_hidden_units_per_area))
    rec_mask_ie_intra_ff = torch.kron(torch.diag(torch.ones(num_areas-1),-1), torch.ones(i_hidden_units_per_area, e_hidden_units_per_area))
    conn_mask['rec_inter_ff'] = torch.cat([
                    torch.cat([rec_mask_ee_intra_ff, rec_mask_ei*0], dim=1),
                    torch.cat([rec_mask_ie_intra_ff, rec_mask_ii*0], dim=1)], dim=0)
    
    # feedback connectivity
    rec_mask_ee_intra_ff = torch.kron(torch.diag(torch.ones(num_areas-1),1), torch.ones(e_hidden_units_per_area, e_hidden_units_per_area))
    rec_mask_ie_intra_ff = torch.kron(torch.diag(torch.ones(num_areas-1),1), torch.ones(i_hidden_units_per_area, e_hidden_units_per_area))
    conn_mask['rec_inter_fb'] = torch.cat([
                    torch.cat([rec_mask_ee_intra_ff, rec_mask_ei*0], dim=1),
                    torch.cat([rec_mask_ie_intra_ff, rec_mask_ii*0], dim=1)], dim=0)


    conn_mask['rec_inter'] = conn_mask['rec']-conn_mask['rec_intra']
    return conn_mask

class EILinear(nn.Module):
    def __init__(self, input_size, output_size, remove_diag, zero_cols_prop, 
                 e_prop, bias, pos_function='abs', conn_mask=None, 
                 init_spectral=None, init_gain=None, balance_ei=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        assert(e_prop<=1 and e_prop>=0)
        
        self.e_prop = e_prop
        self.e_size = int(e_prop * input_size)
        self.i_size = input_size - self.e_size
        self.zero_cols = round(zero_cols_prop * input_size)

        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        sign_mask = torch.FloatTensor([1]*self.e_size+[-1]*self.i_size).reshape(1, input_size)
        exist_mask = torch.cat([torch.ones(input_size-self.zero_cols), torch.zeros(self.zero_cols)]).reshape([1, input_size])

        if conn_mask is None:
            mask = (sign_mask*exist_mask).repeat([output_size, 1])
            self.register_buffer('mask', mask)
        else:
            mask = (sign_mask*exist_mask).repeat([output_size, 1])*conn_mask
            self.register_buffer('mask', mask)
        self.pos_func = _get_pos_function(pos_function)
        if remove_diag:
            assert(input_size==output_size)
            self.mask[torch.eye(input_size)>0.5]=0.0
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(init_spectral, init_gain, balance_ei)

    def reset_parameters(self, init_spectral, init_gain, balance_ei):
        with torch.no_grad():
            # nn.init.uniform_(self.weight, a=0, b=math.sqrt(1/(self.input_size-self.zero_cols)))
            # nn.init.kaiming_uniform_(self.weight, a=1)
            self.weight.data = torch.from_numpy(
                np.random.gamma(np.ones_like(self.mask.numpy()), 
                                np.sqrt(1/(np.abs(self.mask).sum(dim=1, keepdim=True).numpy()+1e-8)), 
                                size=self.weight.data.shape)).float()
            # Scale E weight by E-I ratio
            if balance_ei is not None and self.i_size!=0:
                # self.weight.data[:, :self.e_size] /= self.e_size/self.i_size
                self.weight.data[:, :self.e_size] /= balance_ei

            if init_gain is not None:
                self.weight.data *= init_gain
            if init_spectral is not None:
                self.weight.data *= init_spectral / np.abs(torch.linalg.eigvals(self.effective_weight())).max()

            if self.bias is not None:
                nn.init.zeros_(self.bias)
    
    def effective_weight(self, w=None):
        if w is None:
            return self.pos_func(self.weight) * self.mask
        else:
            return w * self.mask.unsqueeze(0)

    def forward(self, input, w=None):
        # weight is non-negative
        if w is None:
            return F.linear(input, self.effective_weight(), self.bias)
        else:
            result = torch.bmm(self.effective_weight(w), input.unsqueeze(2)).squeeze(2) 
            if self.bias is not None:
                result += self.bias
            return result

# class MaskedLinear(nn.Module):
#     def __init__(self, input_size, output_size, bias, 
#                  conn_mask=None, init_gain=None):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size

#         self.weight = nn.Parameter(torch.empty(output_size, input_size))
#         # No sign constraints - all weights can be positive or negative
#         exist_mask = torch.ones(1, input_size)

#         if conn_mask is None:
#             mask = exist_mask.repeat([output_size, 1])
#             self.register_buffer('mask', mask)
#         else:
#             mask = exist_mask.repeat([output_size, 1]) * conn_mask
#             self.register_buffer('mask', mask)

#         if bias:
#             self.bias = nn.Parameter(torch.empty(output_size))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters(init_gain)

#     def reset_parameters(self, init_gain):
#         with torch.no_grad():
#             # Initialize weights with Gaussian distribution using Glorot-style variance
#             nn.init.kaiming_normal_(self.weight, a=1)
            
#             if init_gain is not None:
#                 self.weight.data *= init_gain

#             if self.bias is not None:
#                 nn.init.zeros_(self.bias)
    
#     def effective_weight(self, w=None):
#         if w is None:
#             return self.weight * self.mask
#         else:
#             return w * self.mask

#     def forward(self, input, w=None):
#         if w is None:
#             return F.linear(input, self.effective_weight(), self.bias)
#         else:
#             result = torch.bmm(self.effective_weight(w), input.unsqueeze(2)).squeeze(2) 
#             if self.bias is not None:
#                 result += self.bias
#             return result


class PlasticSynapse(nn.Module):
    def __init__(self, input_size, output_size, dt_w, tau_w, weight_bound, sigma_w, plas_mask):
        '''
        w(t+1)(rwd, post, pre) = w(t)+kappa*rwd*(hebb(post, pre)+noise)
        '''
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight_bound = weight_bound
        self.lb = torch.zeros(self.output_size, self.input_size)
        self.ub = torch.ones(self.output_size, self.input_size)*weight_bound
        
        self.dt_w = dt_w
        self.tau_w = tau_w
        
        self.alpha_w = dt_w / self.tau_w
        self._sigma_w = np.sqrt(2/self.alpha_w) * sigma_w

        self.kappa = nn.Parameter(torch.ones(self.output_size, self.input_size)*(self.alpha_w))
        self.plas_mask = plas_mask

    def effective_lr(self):
        if self.plas_mask is not None:
            return self.plas_mask*self.kappa.abs()
        else:
            return self.kappa.abs()

    def forward(self, w, baseline, R, pre, post):
        '''
            w: previous plastic weight 
            baseline: fixed weight
            R: rwd \in {-1, +1}
            pre: pre-synaptic firing rates
            post: post-synaptic firing rates
            kappa: learning rate
        '''
        new_w = baseline*(self.alpha_w) + w*(1-self.alpha_w) \
            + R*self.effective_lr()*(torch.bmm(post.unsqueeze(2), pre.unsqueeze(1))+self._sigma_w*torch.randn_like(w))
        new_w = torch.clamp(new_w, self.lb, self.ub)
        return new_w
    

class LeakyRNNCell(nn.Module):
    def __init__(self, input_config, hidden_size, num_areas, 
                activation, dt_x, tau_x, train_init_state,
                e_prop, sigma_rec, sigma_in, init_spectral, 
                balance_ei, conn_mask):
        super().__init__()
        self.input_config = input_config
        self.total_input_size = sum([input_size for (_, (input_size, _)) in input_config.items()])
        self.hidden_size = hidden_size
        self.e_size = int(e_prop * hidden_size)
        self.i_size = self.hidden_size-self.e_size
        self.num_areas = num_areas

        self.x2h = {}
        for (input_name, (input_size, input_target)) in self.input_config.items():
            curr_in_mask = torch.zeros(self.num_areas, 1)
            curr_in_mask[input_target, :] = 1
            self.x2h[input_name] = EILinear(input_size, hidden_size*num_areas, remove_diag=False, pos_function='abs',
                            e_prop=1, zero_cols_prop=0, bias=False, init_gain=math.sqrt(1/hidden_size/len(input_target)),
                            conn_mask=_get_connectivity_mask_in(curr_in_mask, input_size, self.e_size, self.i_size))
        self.x2h = nn.ModuleDict(self.x2h)

        self.h2h = EILinear(hidden_size*num_areas, hidden_size*num_areas, remove_diag=True, pos_function='abs',
                            e_prop=e_prop, zero_cols_prop=0, bias=True, init_gain=1,
                            init_spectral=init_spectral, balance_ei=balance_ei,
                            conn_mask=conn_mask['rec'])
        
        self.tau_x = tau_x
        self.dt_x = dt_x

        self.alpha_x = dt_x / self.tau_x
        self.oneminusalpha_x = 1 - self.alpha_x
        self._sigma_rec = np.sqrt(2*self.alpha_x) * sigma_rec
        self._sigma_in = np.sqrt(2/self.alpha_x) * sigma_in

        if train_init_state:
            self.x0 = nn.Parameter(torch.zeros(1, hidden_size))
        else:
            self.x0 = torch.zeros(1, hidden_size)

        self.activation = _get_activation_function(activation)

    def forward(self, x, state, wh):

        output = self.activation(state)

        total_input = 0
        
        for (input_name, input_x) in x.items():
            input_x = torch.relu(input_x + self._sigma_in * torch.randn_like(input_x))
            total_input += self.x2h[input_name](input_x)

        total_input += self.h2h(output, wh)

        new_state = state * self.oneminusalpha_x + total_input * self.alpha_x + self._sigma_rec * torch.randn_like(state)
        new_output = self.activation(new_state)
        
        return new_state, new_output

class HierarchicalPlasticRNN(nn.Module):
    def __init__(self, input_config, hidden_size, output_config, num_areas,
                inter_regional_sparsity, inter_regional_gain,
                plastic, activation, dt_x, dt_w, tau_x, tau_w, 
                e_prop, sigma_rec, sigma_in, sigma_w, 
                balance_ei=True, init_spectral=None, train_init_state=False, weight_bound=1.0):
        super().__init__()

        self.input_config = input_config
        # input configs contain entries of type {'name': (input_size, destination area)}
        self.hidden_size = hidden_size
        self.output_config = output_config
        # input configs contain entries of type {'name': (output_size, source area)}
        self.num_areas = num_areas
        self.e_size = int(e_prop * hidden_size)
        self.i_size = hidden_size-self.e_size
        self.plastic = plastic
        self.weight_bound = weight_bound

        # specify connectivity
        rec_mask_weight = torch.eye(self.num_areas) + torch.diag(torch.ones(self.num_areas-1), 1) + torch.diag(torch.ones(self.num_areas-1), -1)
        rec_mask_plas = rec_mask_weight

        self.conn_masks = _get_connectivity_mask_rec(rec_mask=rec_mask_weight, num_areas=self.num_areas,
                                                 e_hidden_units_per_area=self.e_size, 
                                                 i_hidden_units_per_area=hidden_size-self.e_size)
        self.plas_mask = _get_plasticity_mask_rec(rec_mask=rec_mask_plas, num_areas=self.num_areas, 
                                                  e_hidden_units_per_area=self.e_size, 
                                                  i_hidden_units_per_area=hidden_size-self.e_size)
        
        self.register_buffer('mask_rec_inter', self.conn_masks['rec_inter'], persistent=False)

        balance_ei = self.conn_masks['rec_intra']\
                    +inter_regional_gain[0]*inter_regional_sparsity[0]*self.conn_masks['rec_inter_ff']\
                    +inter_regional_gain[1]*inter_regional_sparsity[1]*self.conn_masks['rec_inter_fb']\
                    -torch.eye(self.hidden_size*self.num_areas)
        balance_ei = balance_ei[:,:self.e_size*self.num_areas].sum(1, keepdim=True)/balance_ei[:,self.e_size*self.num_areas:].sum(1, keepdim=True)

        self.rnn = LeakyRNNCell(input_config=self.input_config, hidden_size=hidden_size, num_areas=num_areas,
                                activation=activation, dt_x=dt_x, tau_x=tau_x, train_init_state=train_init_state, 
                                e_prop=e_prop, sigma_rec=sigma_rec, sigma_in=sigma_in, init_spectral=init_spectral,
                                balance_ei=balance_ei, conn_mask=self.conn_masks)
        
        self.plasticity = PlasticSynapse(input_size=self.hidden_size*self.num_areas, output_size=self.hidden_size*self.num_areas, 
                                         dt_w=dt_w, tau_w=tau_w, weight_bound=weight_bound, sigma_w=sigma_w, plas_mask=self.plas_mask)
        
        # sparsify inter-regional connectivity, but not enforeced
        sparse_mask_ff = (torch.rand((self.conn_masks['rec_inter_ff'].abs().sum().long(),))<inter_regional_sparsity[0])
        sparse_mask_fb = (torch.rand((self.conn_masks['rec_inter_fb'].abs().sum().long(),))<inter_regional_sparsity[1])
        self.rnn.h2h.weight.data[self.conn_masks['rec_inter_ff'].abs()>0.5] *= sparse_mask_ff*inter_regional_gain[0]
        self.rnn.h2h.weight.data[self.conn_masks['rec_inter_ff'].abs()>0.5] += 1e-8
        self.rnn.h2h.weight.data[self.conn_masks['rec_inter_fb'].abs()>0.5] *= sparse_mask_fb*inter_regional_gain[1]
        self.rnn.h2h.weight.data[self.conn_masks['rec_inter_fb'].abs()>0.5] += 1e-8

        # readout
        self.h2o = {}
        for (output_name, (output_size, output_source)) in self.output_config.items():
            curr_out_mask = torch.zeros(1, self.num_areas)
            curr_out_mask[:,output_source] = 1
            self.h2o[output_name] = EILinear(self.hidden_size*self.num_areas, output_size, remove_diag=False, pos_function='abs',
                            e_prop=1, zero_cols_prop=0, bias=True, init_gain=1,
                            conn_mask = _get_connectivity_mask_out(curr_out_mask, output_size, self.e_size, self.i_size))
        self.h2o = nn.ModuleDict(self.h2o)

        # init state
        if train_init_state:
            self.h0 = nn.Parameter(torch.zeros(1, hidden_size*self.num_areas))
        else:
            self.register_buffer("h0", torch.zeros(1, hidden_size*self.num_areas))

    def init_hidden(self, x):
        batch_size = x['reward'].shape[0]
        h_init = self.h0 + self.rnn._sigma_rec * torch.randn(batch_size, self.hidden_size*self.num_areas, device=x['reward'].device)
        if self.plastic:
            return [h_init, self.rnn.h2h.pos_func(self.rnn.h2h.weight).unsqueeze(0).repeat(batch_size, 1, 1)]
        else:
            return [h_init]

    def forward(self, x, steps, neumann_order=10, 
                hidden=None, w_hidden=None, update_w=False, 
                save_all_states=False):
        # initialize firing rate and fixed weight
        if hidden is None and w_hidden is None:
            hidden, w_hidden = self.init_hidden(x)
        
        if save_all_states: 
            hs = []

        # fixed point iterations, not keeping gradient
        for _ in range(steps-neumann_order):
            with torch.no_grad():
                hidden, output = self.rnn(x, hidden, w_hidden)
            if save_all_states:
                hs.append(hidden)
        # k-order neumann series approximation
        hidden = hidden.detach()
        for _ in range(min(steps, neumann_order)):
            hidden, output = self.rnn(x, hidden, w_hidden)
            if save_all_states:
                hs.append(hidden)

        # if dopamine is not None, update weight
        if update_w is True:
            DAs = (torch.tensor([[-1, 1]])*x['reward']).sum(-1)
            w_hidden = self.plasticity(w_hidden, self.rnn.h2h.pos_func(self.rnn.h2h.weight).unsqueeze(0), DAs, output, output)
        
        if save_all_states:
            hs = torch.stack(hs, dim=0)
        else:
            hs = output

        os = {}
        for output_name in self.h2o.keys():
            os[output_name] = self.h2o[output_name](output)
        return os, hidden, w_hidden, hs
