
import torch
import numpy as np
from contextlib import contextmanager
from easydict import EasyDict as edict

def enumerate_indices(total_samples,selected_samples):
    if selected_samples > 0:
        indices = torch.randperm(total_samples)
        indices = indices[:selected_samples]
    else:
        indices = torch.arange(total_samples)
    return indices

def set_random_state(rng_state):
    torch.set_rng_state(rng_state['th'])
    np.random.set_state(rng_state['np'])
    for device,device_state in enumerate(rng_state['cuda']):
        torch.cuda.set_rng_state(device_state,device)

def get_random_state():
    th_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all()
    np_rng_state = np.random.get_state()
    rng_state = edict({'th':th_rng_state,'np':np_rng_state,
                       'cuda':cuda_rng_state})
    return rng_state

class RandomOnce():
    """
    Is your random dataset resampling new behavior each batch?
    - new noise pattern each sample?
    - new dynamics each sample?

    This Class can be used along with "with" so a random
    seed is re-used for each sample index in a dataset.

    Changes required in the calling objection.

    Now each image gets a single random realization per creation,
    and your random realization doesn't need to be saved.
    """

    def __init__(self,activate,nsamples):
        """

        states : the random states we will use
        activate : do we actually force a single random sample?
        index : used in context manager
        current : used in context manager

        """
        self.activate = activate
        self.states = self._sim_random_states(nsamples)
        self.index = None
        self.current = None

    #
    # -- Managing Random States --
    #

    def _sim_random_states(self,nsamples):
        if self.activate is False: return
        states = [None,]*nsamples
        for i in range(nsamples):
            states[i] = get_random_state()
            np.random.rand(1)
            torch.rand(1)
        return states

    #
    # -- Create Context Manager for Easy Use --
    #
    #   randOnce = RandomOnce(...)
    #   with randOnce.cm(...):
    #       ... do stuff with set seed ...
    #
    #

    @contextmanager
    def set_state(self,index):
        try:
            self.index = index
            yield self._enter(index)
        finally:
            self._exit()

    def _enter(self,index):
        if self.activate:

            # -- save state --
            self.current = get_random_state()

            # -- get fixed state --
            new_state = self.states[self.index]

            # -- set to fixed state --
            set_random_state(new_state)

    def _exit(self):
        if self.activate:
            # -- reset to original state --
            set_random_state(self.current)


