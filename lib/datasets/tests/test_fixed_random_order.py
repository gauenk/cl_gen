
import pytest

# -- python --
import numpy as np
import pandas as pd
from pyutils import create_named_meshgrid,groupby_fields

# -- pytorch --
import torch
from torch.utils.data import DataLoader


# -- create a dataset to return indices --
class IndexDataset():

    def __init__(self,size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self,index):
        rands = torch.randint(10,(3,))
        if self.size > index: return index,rands
        else: raise ValueError(f"index [{index}] too big for size [{self.size}]")



@pytest.fixture
def setup_dataset():
    pass

def loader_rands_grid():
    seed = [123,234]
    batch_size = [10]
    shuffle = [True,False]
    nworkers = [2]
    extra_ops = [0,1]
    items = [seed,batch_size,shuffle,nworkers,extra_ops]
    names = ['seed','batch_size','shuffle','nworkers','extra_ops']
    mesh = pd.DataFrame(create_named_meshgrid(items,names))
    field_names = ['seed','shuffle']
    # field_names = ['shuffle']
    exp_pairs = groupby_fields(field_names,mesh)

    def create_exp_ids(pair,field_names):
        id_str = ""
        for field in field_names:
            id_str += f"{field}:"
            for item in pair:
                id_str += f"{item[field]}+"
            id_str = id_str[:-1]
            id_str += "_"
        id_str = id_str[:-1]        
        return id_str

    names = ['seed','shuffle']
    for i,pair in enumerate(exp_pairs):
        id_str = create_exp_ids(pair,names)
        print(id_str)
        exp_pairs[i] = pytest.param(pair,id=id_str)
    return exp_pairs

def loader_rands(params):

    seed = params.seed
    batch_size = params.batch_size
    shuffle = params.shuffle
    nworkers = params.nworkers
    extra_ops = params.extra_ops

    def set_torch_seed(worker_id):
        pass 
        # torch.manual_seed(torch.initial_seed() + worker_id)
    
    torch.manual_seed(seed)
    dataset = IndexDataset(20)
    loader_kwargs = {'batch_size': batch_size,
                     'shuffle': shuffle,
                     'drop_last':False,
                     'num_workers': nworkers,
                     'pin_memory':True,
                     'worker_init_fn':set_torch_seed}
    loader = DataLoader(dataset,**loader_kwargs)
    nloader = 2#len(loader)
    nbatches = 10
    epochs = 2

    index,rands,opts = [],[],[]
    for epoch in range(epochs):
        index_e,rands_e,opts_e = [],[],[]
        # torch.manual_seed(seed+1+epoch)#seed)
        if extra_ops == 0:
            dont_use = torch.randint(10,(3,))
        torch.manual_seed(seed+1+epoch)#seed)
        train_iter = iter(loader)
        for batch_idx in range(nbatches+2):
            if batch_idx >= nloader:
                init = torch.initial_seed()
                torch.manual_seed(seed+1+epoch+init)#seed)
                train_iter = iter(loader)
            batch = next(train_iter)

            if extra_ops == 0:
                opts_e.append(torch.randint(10,(3,)))
            else:
                dont_use = torch.randint(10,(3,))
                dont_use = torch.randint(10,(3,))
                #print("post-loader",dont_use)
                opts_e.append(torch.zeros((3,)))
            for elem in zip(*batch):
                index_i,rands_i = elem
                index_e.append(index_i)
                rands_e.append(rands_i)
        rands_e = torch.stack(rands_e)
        index_e = torch.stack(index_e)
        opts_e = torch.stack(opts_e)
        rands.append(rands_e)
        index.append(index_e)
        opts.append(opts_e)
    opts = torch.stack(opts)
    rands = torch.stack(rands)
    index = torch.stack(index)
    id_str = f"loader_rands_{seed}_{batch_size}_{shuffle}_{nworkers}"
    return index,rands,opts
"""
Lessons

--- lesson 1

"torch.set_manual" must be immediately before data loader for datasets to be 
(*) a fixed random pattern for a specific seed
    regardless of what happens _inside_ the function

--- lesson 2

"torch.set_manual" must use a changing value (e.g. epoch_num + seed) to have
(*) a different set of random numbers for each epoch
    
--- lesson 3

We don't need to use 'worker_init_fn'. But if we do using
"torch.manual_seed(torch.initial_seed() + worker_id)"
won't break anything

--- lesson 4

If we want to use an iterator instead of the generator, use need 
to reset seed directly before out iter(*) call. For example,

if batch_idx >= nloader:
    init = torch.initial_seed()
    torch.manual_seed(seed+1+epoch+init)#seed)
    train_iter = iter(loader)

"""

@pytest.mark.parametrize('pair',loader_rands_grid())
def test_fixed_ordering(pair):
    index1,rands1,opts1 = loader_rands(pair[0])
    index2,rands2,opts2 = loader_rands(pair[1])
    
    # print(index1)
    # print(rands1)
    # print(opts1)
    # -- ensure selected data is equal --
    diff = torch.sum(torch.abs(index1 - index2)).item()
    assert diff == 0

    # -- ensure randomness is even with non-data random opts  --
    diff = torch.sum(torch.abs(rands1 - rands2)).item()
    assert diff == 0
    
    # -- ensure randomness for non-data opts are equal --
    # diff = torch.sum(torch.abs(opts1 - opts2)).item()
    # assert diff == 0

    # -- ensure each epoch is different --
    shuffle = all([elem.shuffle for elem in pair])
    if shuffle:
        names = ["index1","rands1","index2","rands2"]
        items = [index1,rands1,index2,rands2]
        for name,item in zip(names,items):
            epochs = item.shape[0]
            for i in range(epochs):
                for j in range(epochs):
                    if i >= j: continue
                    # print(f"{name}",i,j)
                    diff = torch.sum(torch.abs(item[i] - item[j])).item()
                    assert diff > 0

