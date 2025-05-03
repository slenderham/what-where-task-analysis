import numpy as np
from jax import jit, vmap
import jax.numpy as jnp
from jax import random
from functools import partial
from jax import lax

key = random.PRNGKey(42)

@jit
def rectanh(x):
    return jnp.maximum(0, jnp.tanh(x))

class LowRankRNN():
    '''
    Simple RNN that is affected by continuous arbitration
    '''
    def __init__(self, input_size, hidden_size, num_populations, cov_dict, 
                 dt_x, tau_x, sigma_rec, sigma_in, exc_ratio):
        self.hidden_size = hidden_size
        self.num_populations = num_populations
        self.cov_dict = cov_dict
        self.exc_ratio = exc_ratio
        self.e_size = int(num_populations*hidden_size*exc_ratio)
        self.i_size = int(num_populations*hidden_size*(1-exc_ratio))
        
        self.tau_x = tau_x
        self.dt_x = dt_x

        self.alpha_x = dt_x / self.tau_x
        self.sigma_rec = np.sqrt(2*self.alpha_x) * sigma_rec
        self.sigma_in = np.sqrt(2/self.alpha_x) * sigma_in

        self.i2h = jnp.zeros((num_populations*hidden_size, num_populations*input_size))
        self.h2h = jnp.zeros((num_populations*hidden_size, num_populations*hidden_size))
        self.h2o = jnp.zeros((num_populations*hidden_size, 1))

        delta_i2h, delta_h2h, delta_h2o = self._add_choice_exec_components()
        self.i2h += delta_i2h / (num_populations*self.hidden_size)**0.5
        self.h2h += delta_h2h / (num_populations*self.hidden_size)**0.5
        self.h2o += delta_h2o / (num_populations*self.hidden_size)**0.5

    def _make_EI_matrix(self, h2h):
        # for simplicity, assume only excitatory population has structure, 
        # and inhibitory population is randomly connected
        min_exc = jnp.min(h2h)

        key, subkey = random.split(key)
        n_exc = random.normal(subkey, (self.e_size))
        n_inh = random.normal(subkey, (self.e_size))

        


    def _add_choice_exec_components(self):
        # sample random vectors for low-rank components

        key, subkey = random.split(key)
        loc_common_subspace = random.normal(subkey, (self.hidden_size,))
        key, subkey = random.split(key)
        stim_common_subspace = random.normal(subkey, (self.hidden_size,))
        
        key, subkey = random.split(key)
        loc_value_subspace = random.normal(subkey, (self.hidden_size,))
        key, subkey = random.split(key)
        stim_value_subspace = random.normal(subkey, (self.hidden_size,))

        key, subkey = random.split(key)
        loc_motor_subspace = random.normal(subkey, (self.hidden_size,))
        key, subkey = random.split(key)
        stim_motor_subspace = random.normal(subkey, (self.hidden_size,))

        key, subkey = random.split(key)
        loc_arbitration_subspace = random.normal(subkey, (self.hidden_size,))
        key, subkey = random.split(key)
        stim_arbitration_subspace = random.normal(subkey, (self.hidden_size,))


        I_loc = loc_value_subspace
        n_loc = self.cov_dict['exec_nI_loc']*loc_value_subspace + \
                    (self.cov_dict['exec_nm_loc']**0.5)*loc_common_subspace
        m_loc = (self.cov_dict['exec_nm_loc']**0.5)*loc_common_subspace + \
                    ((self.cov_dict['exec_nm_loc']-self.cov_dict['exec_nm_loc'])**0.5)*loc_motor_subspace
        w_loc = self.cov_dict['exec_mw_loc']/((self.cov_dict['exec_nm_loc']-self.cov_dict['exec_nm_loc'])**0.5)*loc_motor_subspace
        ctx_loc = self.cov_dict['exec_cc_stim']*loc_arbitration_subspace # in loc context, high variance input to stim population

        I_stim = stim_value_subspace
        n_stim = self.cov_dict['exec_nI_stim']*stim_value_subspace + \
                    (self.cov_dict['exec_nm_stim']**0.5)*stim_common_subspace
        m_stim = (self.cov_dict['exec_nm_stim']**0.5)*loc_common_subspace + \
                    ((self.cov_dict['exec_nm_stim']-self.cov_dict['exec_nm_stim'])**0.5)*stim_motor_subspace
        w_stim = self.cov_dict['exec_mw_stim']/((self.cov_dict['exec_nm_stim']-self.cov_dict['exec_nm_stim'])**0.5)*stim_motor_subspace
        ctx_stim = self.cov_dict['exec_cc_loc']*stim_arbitration_subspace # in stim context, high variance input to loc population

        I_total = jnp.concatenate(
            jnp.concatenate([I_loc, jnp.zeros((self.hidden_size,))], axis=0),
            jnp.concatenate([jnp.zeros((self.hidden_size,)), I_stim], axis=0),
        ) # [2*hidden_size X 2]
        ctx_total = jnp.concatenate(
            jnp.concatenate([jnp.zeros((self.hidden_size,)), ctx_loc], axis=0),
            jnp.concatenate([ctx_stim, jnp.zeros((self.hidden_size,))], axis=0),
        ) # [2*hidden_size X 2]
        n_total = jnp.concatenate([n_loc, n_stim])
        m_total = jnp.concatenate([m_loc, m_stim])
        w_total = jnp.concatenate([w_loc, w_stim]).reshape((1, self.num_populations*self.hidden_size))

        J_total = m_total.reshape((self.num_populations*self.hidden_size, 1))*n_total.reshape((1, self.num_populations*self.hidden_size))

        return jnp.concatenate([I_total, ctx_total], axis=1), J_total, w_total

    def run(self, inputs):
        # inputs: [time_step X 4], 2 for values, 2 for contexts
        # outputs: [num_populations*hidden_size X 1], 
        # hidden state: [num_populations*hidden_size X 1]
        def step(hidden_state, step_input):
            # hidden_state: [num_populations*hidden_size X 1]
            # input: [num_populations*hidden_size X 1]
            hidden_state = (1-self.alpha_x) * hidden_state + \
                self.alpha_x * (self.i2h @ step_input + self.h2h @ rectanh(hidden_state))
            output = self.h2o @ hidden_state
            return hidden_state, output
        
        key, subkey = random.split(key)
        init_state = random.normal(subkey, (self.num_populations*self.hidden_size, 1))*self.sigma_rec
        lax.scan(f=step, init=init_state, xs=inputs)

        return 