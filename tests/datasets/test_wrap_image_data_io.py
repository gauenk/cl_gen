
# -- python --
import random
import numpy as np
import pandas as pd
from easydict import EasyDict as edict

# -- pytorch --
import torch
import torch.nn.functional as F

# -- project --
import settings
from datasets.wrap_image_data import load_image_dataset,load_resample_dataset,sample_to_cuda

def get_cfg_defaults():
    cfg = edict()

    # -- frame info --
    cfg.nframes = 10
    cfg.frame_size = 32

    # -- data config --
    cfg.dataset = edict()
    cfg.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.dataset.name = "voc"
    cfg.dataset.dict_loader = True
    cfg.dataset.num_workers = 2
    cfg.set_worker_seed = True
    cfg.batch_size = 1
    cfg.drop_last = {'tr':True,'val':True,'te':True}
    cfg.noise_params = edict({'pn':{'alpha':10.,'std':0},
                              'g':{'std':25.0},'ntype':'g'})
    cfg.dynamic_info = edict()
    cfg.dynamic_info.mode = 'global'
    cfg.dynamic_info.frame_size = cfg.frame_size
    cfg.dynamic_info.nframes = cfg.nframes
    cfg.dynamic_info.ppf = 0
    cfg.dynamic_info.textured = True

    cfg.random_seed = 0

    # -- combo config --
    cfg.nblocks = 3
    cfg.patchsize = 3
    # cfg.score_fxn_name = "bootstrapping"
    cfg.score_fxn_name = "bootstrapping_mod2"
    # cfg.score_fxn_name = "bootstrapping_mod3"
    # cfg.score_fxn_name = "bootstrapping_mod4"

    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

def batch_dim0(sample):
    dim1 = ['burst','noisy','res','clean_burst','sburst','snoisy']    
    skeys = list(sample.keys())
    for field in dim1:
        if not(field in skeys): continue
        sample[field] = sample[field].transpose(1,0)

def convert_keys(sample):
    translate = {'noisy':'dyn_noisy',
                 'burst':'dyn_clean',
                 'snoisy':'static_noisy',
                 'sburst':'static_clean',
                 'flow':'flow_gt',
                 'index':'image_index'}

    for field1,field2 in translate.items():
        sample[field2] = sample[field1]
        del sample[field1]
    return sample

def sample_new_bursts(cfg,nbatches=5):
    
    # -- set random seed --
    set_seed(cfg.random_seed)	

    # -- load dataset --
    print("load image dataset.")
    data,loaders = load_image_dataset(cfg)
    image_iter = iter(loaders.tr)    

    # -- init stores --
    records = edict({'rng_state':[],'tl_index':[],'image_index':[],
                     'dyn_noisy':[],'dyn_clean':[],'flow_gt':[],
                     'static_noisy':[],'static_clean':[]})

    # -- sample data --
    for image_bindex in range(nbatches):

        # -- sample & unpack batch --
        sample = next(image_iter)
        # sample_to_cuda(sample)
        dyn_noisy = sample['noisy'] # dynamics and noise
        dyn_clean = sample['burst'] # dynamics and no noise
        static_noisy = sample['snoisy'] # no dynamics and noise
        static_clean = sample['sburst'] # no dynamics and no noise
        flow_gt = sample['flow']
        image_index = sample['index']
        tl_index = sample['tl_index']
        rng_state = sample['rng_state']

        #
        # -- append to records --
        #

        # -- unpack according to batchsize --
        for b in range(cfg.batch_size):
            records['rng_state'].append(rng_state[b])
            records['tl_index'].append(tl_index[b])
            records['image_index'].append(image_index[b])
            records['dyn_noisy'].append(dyn_noisy[:,b])
            records['dyn_clean'].append(dyn_clean[:,b])
            records['flow_gt'].append(flow_gt[b])
            records['static_noisy'].append(static_noisy[:,b])
            records['static_clean'].append(static_clean[:,b])
        print(image_bindex,records['tl_index'])
    
    print("-"*10)
    print(len(records['image_index']))
    return records
        
def reload_samples(cfg,records,nbatches):

    # -- set random seed --
    set_seed(cfg.random_seed)	

    # -- init summary results --
    results = dict.fromkeys(records.keys(),None)
    for field in list(results.keys()):
        results[field] = []
        if not torch.is_tensor(records[field][0]):
            del results[field]
    
    # -- load dataset --
    print("load image dataset.")
    # nbatches = len(records['image_index'])
    records_tr = {'tl_index':records.tl_index,
                  'image_index':records.image_index,
                  'rng_state':records.rng_state,}
    named_records = edict({'tr':records_tr,'val':records_tr,'te':records_tr})
    data,loaders = load_resample_dataset(cfg,named_records)
    train_iter = iter(loaders.tr)

    # -- sample data --
    records_image_index = np.stack([ii.numpy() for ii in records['image_index']])
    for batch_count in range(nbatches):

        print("-="*30+"-")
        print(f"Running image batch index: {batch_count}")
        print("-="*30+"-")

        sample = next(train_iter)
        batch_dim0(sample)
        convert_keys(sample)

        # -- compare with groundtruth sample --
        for batch_index in range(cfg.batch_size):
            image_index = int(sample['image_index'][batch_index])
            record_index = np.where(records_image_index == image_index)[0][0]
            for field in records.keys():
                record_field = records[field][record_index]
                if torch.is_tensor(sample[field]):
                    record_field = record_field.type(torch.float)
                    sample_field = sample[field][batch_index].type(torch.float)
                    loss = F.mse_loss(sample_field,record_field).item()
                    results[field].append(loss)
                    print(field,loss)

    # -- print summary --
    print("-="*30+"-")
    print("Summary")
    print("-="*30+"-")
    for field in results.keys():
        print(field,np.mean(results[field]))

    return results
    

def test_load_from_params():

    # -- run exp --
    cfg = get_cfg_defaults()
    cfg.random_seed = 123
    nbatches = 20
    records = sample_new_bursts(cfg,nbatches)
    results = reload_samples(cfg,records,nbatches)
    
    # -- test --
    print("-="*30+"-")
    print("Final Tests.")
    print("-="*30+"-")
    for field in results.keys():
        assert np.isclose(np.mean(results[field]),0),f"Difference detected in field [{field}]"



