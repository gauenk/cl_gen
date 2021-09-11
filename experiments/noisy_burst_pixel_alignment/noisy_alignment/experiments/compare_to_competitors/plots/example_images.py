# -- python imports --
import numpy as np
from easydict import EasyDict as edict
from einops import rearrange

# -- plotting imports --
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt

# -- project imports --
from datasets.wrap_image_data import load_resample_dataset,sample_to_cuda
# from datasets import load_dataset,load_resample_dataset,sample_to_cuda

def info_from_records(records):
    image_index = records['image_index']
    rng_state = [r[0] for r in records['rng_state']]
    info = {'image_index':image_index,'rng_state':rng_state}
    return info

def get_record_from_iindex(records,image_index):
    image_index = int(image_index)
    records_image_index = np.array(records['image_index'])
    record_index = np.where(records_image_index == image_index)[0][0]
    record = {field:records[field][record_index] for field in records.keys()}
    return record

def plot_example_images(records,exp_cfgs):

    # -- Plot per Experiment --
    for exp_cfg in exp_cfgs:
        exp_records = records[records['uuid'] == exp_cfg.uuid]
        plot_example_images_experiment(exp_cfg,exp_records)
        break
        
def plot_example_images_experiment(cfg,records):

    # -- init info --
    records = records.to_dict('list')
    print(records.keys())
    print(records['image_index'])
    info = info_from_records(records)
    named_info = edict({'tr':info,'val':info,'te':info})
    cfg.shuffle_dataset = False
    cfg.frame_size = [256,256]
    data,loaders = load_resample_dataset(cfg,named_info,use_wrapper=False)
    nsamples = len(records['image_index'])

    # -- loop over samples --
    for sindex in range(nsamples):
        sample = data.tr[sindex] # we use "data" so no batch dimension

        # -- get image index --
        if 'index' in sample: index = sample['index']
        elif 'image_index' in sample: index = sample['image_index']
        else: raise KeyError("Where is the image index key??")
        record = get_record_from_iindex(records,index)
        print(list(record.keys()))
        
        # -- get clean images --
        if 'clean' in sample: clean = sample['clean']
        elif 'dyn_clean' in sample: clean = sample['dyn_clean']
        else: raise KeyError("Where is the clean burst key??")

        # -- get noisy images --
        if 'noisy' in sample: noisy = sample['noisy']
        elif 'dyn_noisy' in sample: noisy = sample['dyn_noisy']
        else: raise KeyError("Where is the noisy burst key??")
        
        # -- create figures --
        plot_image_burst(noisy,"noisy")
        plot_image_burst(clean,"clean")

        break

def plot_image_burst(burst,name):
    nframes = burst.shape[0]
    fig,ax = plt.subplots(1,nframes,figsize=(8,8))
    burst = rearrange(burst,'t c h w -> t h w c')
    for t in range(nframes):
        ax[t].imshow(burst[t])
    fn = f"./example_sample_{name}.png"
    plt.savefig(fn,transparent=True,dpi=300)
    print(f"Saved example sample image at location [{fn}]")
    plt.clf()
    plt.cla()
    plt.close("all")
    
