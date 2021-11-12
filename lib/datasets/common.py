
# -- python imports --
import numpy as np
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def return_optional(pydict,field,default):
    if pydict is None: return default
    if field in pydict.keys(): return pydict[field]
    else: return default

def get_loader(cfg,data,batch_size,mode=None):
    if not(mode is None):
        print("WARNING: [mode] in get_loader is depcrecated.")
    use_ddp = return_optional(cfg,"use_ddp",False)
    if use_ddp:
        loader = get_loader_ddp(cfg,data)
    else:
        loader = get_loader_serial(cfg,data,batch_size)
    return loader


def get_loader_serial(cfg,data,batch_size):
    # -- default for non-compat configs --
    if 'drop_last' in cfg.keys(): drop_last = edict(cfg.drop_last)
    else: drop_last = edict({'tr':True,'val':True,'te':True})

    num_workers = return_optional(cfg,"num_workers",0)
    shuffle = return_optional(cfg,"shuffle_dataset",True)
    loader_kwargs = {'batch_size': batch_size,
                     'shuffle':shuffle,
                     'drop_last':True,
                     'num_workers':num_workers,
                     'pin_memory':True}
    set_worker_seed = return_optional(cfg,"set_worker_seed",False)
    if set_worker_seed:
        loader_kwargs['worker_init_fn'] = set_torch_seed

    use_collate = return_optional(cfg,"use_collate",True)
    triplet_loader = return_optional(cfg,"triplet_loader",False)
    dict_loader = return_optional(cfg,"dict_loader",True)
    if use_collate:
        if triplet_loader:
            loader_kwargs['collate_fn'] = collate_triplet_fn
        elif cfg.dataset.dict_loader:
            loader_kwargs['collate_fn'] = collate_dict
        else:
            loader_kwargs['collate_fn'] = collate_fn

    loader = edict()
    loader.tr = DataLoader(data.tr,**loader_kwargs)
    loader_kwargs['drop_last'] = drop_last.val
    loader.val = DataLoader(data.val,**loader_kwargs)
    loader_kwargs['shuffle'] = shuffle
    loader_kwargs['drop_last'] = drop_last.te
    loader.te = DataLoader(data.te,**loader_kwargs)
    return loader

def get_loader_ddp(cfg,data):
    loader = edict()
    loader_kwargs = {'batch_size':cfg.batch_size,
                     'shuffle':False,
                     'drop_last':True,
                     'num_workers': 1, #cfg.num_workers,
                     'pin_memory':True}
    if cfg.use_collate:
        loader_kwargs['collate_fn'] = collate_fn

    ws = cfg.world_size
    r = cfg.rank
    loader = edict()

    sampler = DistributedSampler(data.tr,num_replicas=ws,rank=r)
    loader_kwargs['sampler'] = sampler
    loader.tr = DataLoader(data.tr,**loader_kwargs)

    del loader_kwargs['sampler']
    loader_kwargs['drop_last'] = False

    # sampler = DistributedSampler(data.val,num_replicas=ws,rank=r)
    # loader_kwargs['sampler'] = sampler
    loader.val = DataLoader(data.val,**loader_kwargs)

    # sampler = DistributedSampler(data.te,num_replicas=ws,rank=r)
    # loader_kwargs['drop_last'] = False
    # loader_kwargs['sampler'] = sampler
    loader.te = DataLoader(data.te,**loader_kwargs)

    return loader

def collate_dict(batch):

    # -- aggregate tensors --
    fbatch = {}
    for sample in batch:
        keys = sample.keys()
        for key,elem in sample.items():
            if not (key in fbatch): fbatch[key] = []
            fbatch[key].append(elem)

    # -- shape tensors --
    dim1 = ['burst','noisy','res','clean_burst','sburst','snoisy']
    dim1 += ['dyn_clean','dyn_noisy','static_clean','static_noisy']
    dim1 += ['nnf','nnf_locs','nnf_vals']
    for key,elem in fbatch.items():
        if key in dim1:
            fbatch[key] = torch.stack(elem,dim=1)
        else:
            if torch.is_tensor(elem[0]):
                fbatch[key] = torch.stack(elem,dim=0)
    return fbatch

def collate_fn(batch):
    noisy,clean = zip(*batch)
    noisy = torch.stack(noisy,dim=1)
    clean = torch.stack(clean,dim=0)
    return noisy,clean

def collate_triplet_fn(batch):
    noisy,res,clean,directions = zip(*batch)
    noisy = torch.stack(noisy,dim=1)
    res = torch.stack(res,dim=1)
    clean = torch.stack(clean,dim=0)
    directions = torch.stack(directions,dim=0)
    return noisy,res,clean,directions

def set_torch_seed(worker_id):
    # np.random.seed(torch.initial_seed() + worker_id)
    torch.manual_seed(torch.initial_seed() + worker_id)
    torch.cuda.manual_seed(torch.initial_seed() + worker_id)

def sample_to_cuda(sample):
    for key in sample.keys():
        if torch.is_tensor(sample[key]):
            sample[key] = sample[key].cuda(non_blocking=True)

def dict_to_device(sample,device):
    for key in sample.keys():
        if torch.is_tensor(sample[key]):
            sample[key] = sample[key].to(device,non_blocking=True)

