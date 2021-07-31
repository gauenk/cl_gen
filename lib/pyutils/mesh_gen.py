"""
A generator of a meshgrid for memory efficient batching

"""
import torch
import numpy as np
import itertools
from einops import rearrange
from easydict import EasyDict as edict
from numba import jit,njit,prange
from numba import typed as nba_typed

def vprint(*args,**kwargs):
    verbose = False
    if verbose:
        print(*args,**kwargs)

@njit
def numba_arange_ranges(ranges):
    expanded = nba_typed.List()
    for t in prange(len(ranges)):
        e = np.arange(ranges[t])
        expanded.append(e)
    return expanded 

def compute_indexing_ranges(r_sizes,sizes):
    nframes = len(r_sizes)
    if hasattr(r_sizes[0],'__len__'):
        r_sizes = [len(x) for x in r_sizes]
    s_sizes = [ r//s + (r % s != 0) for r,s in zip(r_sizes,sizes) ]
    return s_sizes

def create_indexing_mesh(r_sizes,sizes):
    """
    s_sizes are the ranges for the meshgrid
    which can be used to index the original
    ranges, each with length r_sizes
    
    s_sizes tells us if we "skip the original ranges 
    of sizes r_sizes using the integers in sizes"
    then we will cover the original meshgrid
    """
    s_sizes = compute_indexing_ranges(r_sizes,sizes)
    s_sizes_slice = [slice(None,s) for s in s_sizes]
    smesh = np.mgrid[s_sizes_slice]
    smesh = np.c_[[x.ravel() for x in smesh]].T
    return smesh,s_sizes

def apply_indexing_mesh(ranges,lmesh,lsizes,index,device=None):
    """
    ranges
        shape = (number of frames, range of each frame)
    lmesh
        shape = (number of sub-meshgrids, number of frames)
        
    sub_mesh
        The subset of the "ranges" meshgrid using the indexed "lmesh" 
    
    """
    if not hasattr(ranges[0],'__len__'):
        #ranges = preallocated_ranges(ranges,allocated.ranges)
        ranges = np.array(ranges)
        ranges = numba_arange_ranges(ranges)
    nframes = len(ranges)
    index_ranges = lmesh[index]
    sub_ranges = []
    for t in range(nframes):
        start_t = index_ranges[t] * lsizes[t]
        end_t = start_t + lsizes[t]
        sub_ranges.append(ranges[t][start_t:end_t])
    sub_mesh = np.c_[[x.ravel() for x in np.meshgrid(*sub_ranges)]].T
    return sub_mesh



# -- now we "recurse" --
def select_indexing_sizes(f_sizes,l1_sizes,K=3,H=2):
    nframes = len(f_sizes)
    if hasattr(f_sizes[0],'__len__'):
        f_sizes = [len(f) for f in f_sizes]
    l2_sizes = np.ones(nframes,dtype=np.int)
    r_sizes = np.array([f//s + (f % s != 0) for f,s in zip(f_sizes,l1_sizes)])
    topK = r_sizes.argsort()[::-1][:K] # topK in numpy
    for k in topK: l2_sizes[k] = H
    l2_sizes = l2_sizes.astype(int)
    return l2_sizes

def recursive_apply_index(mesh,sizes,ranges,index,device=None):
    nlevels = len(sizes)
    if nlevels == 2:
        sub_mesh = apply_indexing_mesh(ranges[-2],mesh,sizes[-1],index,device)
        # print("[yield] sub_mesh.shape[0]", sub_mesh.shape[0])
        yield sub_mesh
    else:
        sub_mesh = apply_indexing_mesh(ranges[-2],mesh,sizes[-1],index)
        # print("[nlevels] sub_mesh.shape[0]", sub_mesh.shape[0], len(ranges))
        for sub_index in range(sub_mesh.shape[0]):
            yield from recursive_apply_index(sub_mesh,sizes[:-1],
                                             ranges[:-1],sub_index,device)

            
class BatchGen():
    def __init__(self,gens,nframes,max_range,device=None):
        self.gens = gens
        self.device = device
        self.nimages = len(gens)
        self.nsegs = len(gens[0])
        self.batch_size = 64
        self.nframes = nframes
        self.max_range = max_range

    def __len__(self):
        exh_search = self.max_range**(self.nframes-1)
        nbatches = exh_search // self.batch_size + 1
        return nbatches

    def __iter__(self):
        return self

    def __next__(self):
        samples = [[[]]]
        while len(samples[0][0]) < self.batch_size:
            sample = self._sample()
            if sample is None: break
            if len(samples[0][0]) > 0:
                samples = torch.cat([samples,sample],dim=2)
            else:
                samples = sample
        if len(samples[0][0]) == 0:
            raise StopIteration
        return samples

    def _sample(self):
        samples = []
        device = self.device
        for i in range(self.nimages):
            samples_i = []
            for s in range(self.nsegs):
                gen = self.gens[i][s]
                res = self._peek(gen)
                if res is None: return None
                sample,ngen = res
                sample = torch.LongTensor(sample).to(device,non_blocking=True)
                # print("sample.shape ",sample.shape)
                # sample = rearrange(sample,'t a -> a t')
                samples_i.append(sample)
            samples_i = torch.stack(samples_i,dim=0)
            samples.append(samples_i)
        samples = torch.stack(samples,dim=0)
        return samples

    def _peek(self,iterable):
        try:
            first = next(iterable)
        except StopIteration:
            return None
        return first, itertools.chain([first],iterable)

def gen_indexing_mesh_levels(ranges,levels_K,levels_H,device=None):
    
    # -- shapes --
    nranges = len(ranges)
    nlevels = len(levels_K)
    
    # -- possible termination --
    if nlevels == 0:
        return np.c_[[x.ravel() for x in np.meshgrid(*ranges)]].T
    
    # -- compute sizes and ranges for each level --
    l_ranges = ranges
    l_sizes = np.ones(nranges,dtype=np.int)
    level_ranges,level_sizes = [ranges],[l_sizes]
    for level in range(nlevels):
        l_sizes = select_indexing_sizes(level_ranges[-1],level_sizes[-1],levels_K[level],levels_H[level])
        l_ranges = compute_indexing_ranges(level_ranges[-1],l_sizes)
        level_sizes.append(l_sizes) # size of each incriment for each frame.
        level_ranges.append(l_ranges) # num. of incriments for each frame
        if np.all([r == 1 for r in l_ranges]): 
            vprint(f"[indexing_mesh] Cut-off at [{level}] levels.")
            level_sizes.append(l_sizes),level_ranges.append(l_ranges)
            break
    #lm1_ranges = level_ranges[-2]
    mesh,_ = create_indexing_mesh(level_ranges[-2],level_sizes[-1])
    verbose = False
    if verbose:
        print("nlevels: ",nlevels)
        print("sizes: ",level_sizes)
        print("ranges: ",level_ranges[1:])
        for level in range(1,nlevels):
            lnumel = np.product(level_ranges[level])
            print(f"Size of mesh @ level [{level}]: {lnumel}")
        print("mesh.shape[0] ",mesh.shape[0])
    
    # -- recursive generator of mesh --
    for index in range(mesh.shape[0]):
        yield from recursive_apply_index(mesh,level_sizes,level_ranges,index,device)
