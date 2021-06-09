
# -- python imports --
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_loader(cfg,data,batch_size,mode):
    if cfg.use_ddp:
        loader = get_loader_ddp(cfg,data)
    else:
        loader = get_loader_serial(cfg,data,batch_size,mode)
    return loader


def get_loader_serial(cfg,data,batch_size,mode):
    # -- default for non-compat configs --
    if 'drop_last' in cfg.keys(): drop_last = edict(cfg.drop_last)
    else: drop_last = edict({'tr':True,'val':True,'te':True})

    loader_kwargs = {'batch_size': batch_size,
                     'shuffle':True,
                     'drop_last':True,
                     'num_workers':cfg.num_workers,
                     'pin_memory':True}
    if cfg.set_worker_seed:
        loader_kwargs['worker_init_fn'] = set_torch_seed
    if cfg.use_collate:
        if cfg.dataset.triplet_loader:
            loader_kwargs['collate_fn'] = collate_triplet_fn
        elif cfg.dataset.dict_loader:
            loader_kwargs['collate_fn'] = collate_dict
        else:
            loader_kwargs['collate_fn'] = collate_fn

    loader = edict()
    loader.tr = DataLoader(data.tr,**loader_kwargs)
    loader_kwargs['drop_last'] = drop_last.val
    loader.val = DataLoader(data.val,**loader_kwargs)
    loader_kwargs['shuffle'] = True
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
    for key,elem in fbatch.items():
        if key in dim1:
            fbatch[key] = torch.stack(elem,dim=1)
        else:
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
    torch.manual_seed(torch.initial_seed() + worker_id)


