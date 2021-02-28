"""
Write an LMDB of PASCAL VOC Data

"""

# -- python imports --
import lmdb,pickle,os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import numpy.random as npr
from easydict import EasyDict as edict

# -- project imports --
from n2sim.sim_search import compute_similar_bursts,kIndexPermLMDB


def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False

def overwite_lmdb(lmdb_path,metadata_fn):
    answer = yes_or_no("Overwrite Existing LMDB?")
    if answer is False:
        print("Completed!")
        return False
    else:
        print("Overwriting LMDB")
        data_file = lmdb_path / Path("./data.mdb")
        lock_file = lmdb_path / Path("./lock.mdb")
        if data_file.exists(): os.remove(data_file)
        if lock_file.exists(): os.remove(lock_file)
        if metadata_fn.exists(): os.remove(metadata_fn)
        lmdb_path.rmdir()
        return True

def create_lmdb_epochs(cfg,dataset,ds_split,epochs=3,maxNumFrames=10,numSim=8,patchsize=3):
    print("Creating Data for {} Epochs".format(epochs))
    for epoch in range(epochs):
        id_str = str(epoch)
        create_lmdb(cfg,dataset,ds_split,maxNumFrames=10,numSim=8,patchsize=3,id_str=id_str)
    print("Completed All Epochs")

def create_lmdb(cfg,dataset,ds_split,epochs=1,maxNumFrames=10,numSim=8,patchsize=3,id_str="all"):

    # -- check some configs --
    assert cfg.noise_type == 'g', "Noise must be Gaussian here"
    assert np.isclose(cfg.noise_params['g']['stddev'],25.), "Noise Level must be 25."
    assert cfg.N == maxNumFrames, "Dataset must load max number of frames."
    assert cfg.dynamic.ppf == 1, "Movement must be one pixel per frame."
    assert cfg.batch_size == 1, "Batch Size must be 1"

    # -- get noise level --
    noise_level = 0
    if cfg.noise_type == 'g': noise_level = int(cfg.noise_params['g']['stddev'])
    else: raise ValueError(f"Unknown Noise Type [{cfg.noise_type}]")
    
    # -- configs --
    num_images = epochs*len(dataset)

    # -- create target path --
    lower_name = cfg.dataset.name.lower()
    ds_path = Path(cfg.dataset.root) / lower_name / Path("./lmdbs")
    if not ds_path.exists(): ds_path.mkdir(parents=True)

    # -- create config strings --
    noise_str = "{}{}".format(cfg.noise_type,noise_level)
    sim_shuffle = "randPerm"
    nf_str = "nf{}".format(maxNumFrames)

    # -- create file names --

    # -- old --
    # lmdb_path = ds_path / Path("./noisy_burst_{}_{}_nf10_ns8_ps3_ppf1_fs128_{}_lmdb_{}".format(ds_split,noise_str,sim_shuffle,id_str))
    # metadata_fn = lmdb_path / Path("./noisy_burst_{}_{}_nf10_ns8_ps3_ppf1_fs128_{}_metadata_{}.pkl".format(ds_split,noise_str,sim_shuffle,id_str))
    lmdb_path = ds_path / Path("./noisy_burst_xburst_{}_{}_{}_ns8_ps3_ppf1_fs128_{}_lmdb_{}".format(ds_split,noise_str,nf_str,sim_shuffle,id_str))
    metadata_fn = lmdb_path / Path("./noisy_burst_xburst_{}_{}_{}_ns8_ps3_ppf1_fs128_{}_metadata_{}.pkl".format(ds_split,noise_str,nf_str,sim_shuffle,id_str))
    if lmdb_path.exists() and not overwite_lmdb(lmdb_path,metadata_fn): return
        

    # -- compute bytes per entry --
    burst, res_imgs, raw_img, directions = dataset[0]
    burst = burst.to(cfg.gpuid)
    sim_burst = compute_similar_bursts(cfg,burst.unsqueeze(1),numSim,patchsize=3,shuffle_k=True,pick_2=True)
    burst_nbytes = burst.cpu().numpy().astype(np.float32).nbytes
    simburst_nbytes = sim_burst.cpu().numpy().astype(np.float32).nbytes
    rawimg_nbytes = raw_img.numpy().astype(np.float32).nbytes
    dirc_nbytes = directions.numpy().astype(np.float32).nbytes
    data_size = (burst_nbytes + simburst_nbytes + rawimg_nbytes + dirc_nbytes) * num_images
    data_mb,data_gb = data_size/(1000.**2.),data_size/(1000.**3)
    print( "%2.2f MB | %2.2f GB" % (data_mb,data_gb) )

    # -- open lmdb file & open writer --
    print(f"Writing LMDB Path: {lmdb_path}")
    env = lmdb.open( str(lmdb_path) , map_size=data_size*1.5)
    txn = env.begin(write=True)

    # -- start lmdb writing loop --
    lmdb_index = 0
    tqdm_iter = tqdm(enumerate(range(num_images)), total=num_images, leave=False)
    commit_interval = 3 

    # -- load cached randperms --
    kindex_ds = kIndexPermLMDB(cfg.batch_size,maxNumFrames)

    for index, key in tqdm_iter:

        # -- write update --
        assert index == key, "These are not the same?"
        tqdm_iter.set_description('Write {}'.format(key))

        # -- load sample --
        burst, res_imgs, raw_img, directions = dataset[index]
        burst = burst.to(cfg.gpuid)
        kindex = kindex_ds[index].to(cfg.gpuid)
        sim_burst = compute_similar_bursts(cfg,burst.unsqueeze(1),numSim,
                                           patchsize=3,shuffle_k=True,
                                           kindex=kindex,pick_2=True)
        burst = burst.cpu()
        sim_burst = sim_burst.cpu()

        # -- sample to numpy --
        burst = burst.numpy()
        raw_img = raw_img.numpy()
        directions = directions.numpy()
        sim_burst = sim_burst.numpy()
        
        # -- create keys for lmdb --
        key_burst = "{}_burst".format(lmdb_index).encode('ascii')
        key_sim_burst = "{}_sim_burst".format(lmdb_index).encode('ascii')
        key_raw = "{}_raw".format(lmdb_index).encode('ascii')
        key_direction = "{}_direction".format(lmdb_index).encode('ascii')
        lmdb_index += 1
        
        # -- add to buffer to write for lmdb --
        txn.put(key_burst, burst)
        txn.put(key_sim_burst, sim_burst)
        txn.put(key_raw, raw_img)
        txn.put(key_direction, directions)

        # -- write to lmdb --
        if (index + 1) % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)

    # -- final write to lmdb & close --
    txn.commit()
    env.close()
    print('Finish writing lmdb')

    # -- write meta info to pkl --
    meta_info = {"num_samples": num_images,
                 "num_sim": numSim,
                 "num_frames":maxNumFrames,
                 "patch_size": patchsize}
    pickle.dump(meta_info, open(metadata_fn, "wb"))

    # -- done! --
    print('Finish creating lmdb meta info.')
    

    
    
