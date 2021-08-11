# -- python imports --
import numpy as np
from easydict import EasyDict as edict

# -- plotting imports --
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt

# -- project imports --
from datasets.wrap_image_data import load_resample_dataset,sample_to_cuda

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
    data,loaders = load_resample_dataset(cfg,named_info)
    nsamples = len(records['image_index'])

    # -- loop over samples --
    for sindex in range(nsamples):
        sample = data.tr[sindex]

        record = get_record_from_iindex(records,sample['index'])
        print(list(record.keys()))
        
        # -- get images -
        noisy = sample['noisy']
        nframes = noisy.shape[0]
        
        # -- create figure --
        fig,ax = plt.subplots(1,nframes,figsize=(8,8))
        for t in range(nframes):
            ax[t].imshow(noisy[t][0])
        plt.savefig("./sample.png",transparent=True,dpi=300)
        break
