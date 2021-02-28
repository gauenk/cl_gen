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

# -- pytorch imports --
import torch

# -- project imports --
from n2sim.sim_search import compute_similar_bursts

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

def create_lmdb(ds_size=10000,epochs=1,maxNumFrames=12,numSim=8,gpuid=2):


    # -- set some configs
    N,C,H,W = maxNumFrames,3,128,128
    K  = numSim
    E = epochs
    D = ds_size
    num_samples = E*D

    # -- create target path --
    ds_path = Path("./data/rand_perms/lmdbs/")
    if not ds_path.exists(): ds_path.mkdir(parents=True)

    # -- create file names --
    lmdb_path = ds_path / Path("./randperm_c{}_hw{}_nf{}_ns{}_e{}_d{}".format(C,H,N,K,E,D))
    metadata_fn = lmdb_path / Path("./metadata.pkl")
    if lmdb_path.exists() and not overwite_lmdb(lmdb_path,metadata_fn): return
        

    # -- compute bytes per entry --    
    perms = []
    for n in range(N): perms.append(torch.stack([torch.randperm(K+1) for _ in range(C*H*W)]).t().long())
    perms = torch.stack(perms).numpy().astype(np.uint8)

    # -- compute total dataset size --
    data_size = perms.nbytes * num_samples
    data_mb,data_gb = data_size/(1000.**2.),data_size/(1000.**3)
    print( "%2.2f MB | %2.2f GB" % (data_mb,data_gb) )

    # -- open lmdb file & open writer --
    print(f"Writing LMDB Path: {lmdb_path}")
    env = lmdb.open( str(lmdb_path) , map_size=data_size*1.5)
    txn = env.begin(write=True)

    # -- start lmdb writing loop --
    lmdb_index = 0
    tqdm_iter = tqdm(enumerate(range(num_samples)), total=num_samples, leave=False)
    commit_interval = 3 
    for index, key in tqdm_iter:

        # -- write update --
        assert index == key, "These are not the same?"
        tqdm_iter.set_description('Write {}'.format(key))

        # -- load sample --
        perms = []
        for n in range(N): perms.append(torch.stack([torch.randperm(K+1) for _ in range(C*H*W)]).t().long())
        perms = torch.stack(perms).numpy().astype(np.uint8)
        
        # -- create keys for lmdb --
        key_perm = "{}_perm".format(lmdb_index).encode('ascii')
        lmdb_index += 1
        
        # -- add to buffer to write for lmdb --
        txn.put(key_perm, perms)

        # -- write to lmdb --
        if (index + 1) % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)

    # -- final write to lmdb & close --
    txn.commit()
    env.close()
    print('Finish writing lmdb')

    # -- write meta info to pkl --
    meta_info = {"num_samples": num_samples,
                 "num_sim": numSim,
                 "num_frames":maxNumFrames,
                 }
    pickle.dump(meta_info, open(metadata_fn, "wb"))

    # -- done! --
    print('Finish creating lmdb meta info.')
    

    
    
