# -- python imports --
import pickle,lmdb
import numpy as np
from pathlib import Path
import numpy.random as npr
from easydict import EasyDict as edict
from einops import rearrange, repeat, reduce

# -- faiss imports --
import faiss_mod
import faiss
import faiss.contrib.torch_utils

# -- pytorch imports --
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms.functional as tvF

# -- project imports --
from pyutils.timer import Timer
from numba_search import search_raw_array_numba

# -- [local] image comparisons --
from .nearest_search import search_raw_array_pytorch,search_mod_raw_array_pytorch
from .prepare_bsd400_lmdb import compute_sim_images,shift_concat_image

def compare_sim_images_methods(cfg,burst,K,patchsize=3):
    b = 0
    n = 0
    ps = patchsize
    img = burst[n,b]
    single_burst = img.unsqueeze(0).unsqueeze(0)

    img_rs = rearrange(img,'c h w -> h w c').cpu()
    t_1 = Timer()
    t_1.tic()
    sim_burst_v1 = rearrange(torch.tensor(compute_sim_images(img_rs, patchsize, K)),'k h w c -> k c h w')
    t_1.toc()

    t_2 = Timer()
    t_2.tic()
    sim_burst_v2 = compute_similar_bursts(cfg,single_burst,K,patchsize=3,shuffle_k=False)[0,0,1:]
    t_2.toc()

    print(t_1,t_2)
    
    print("v1",sim_burst_v1.shape)
    print("v2",sim_burst_v2.shape)
    for k in range(K):
        print("mse-v1-{}".format(k),F.mse_loss(sim_burst_v1[k].cpu(),img.cpu()))
        print("mse-v2-{}".format(k),F.mse_loss(sim_burst_v2[k].cpu(),img.cpu()))
        print("mse-{}".format(k),F.mse_loss(sim_burst_v1[k].cpu(),sim_burst_v2[k].cpu()))

def compare_sim_patches_methods(cfg,burst,K,patchsize=3):
    b = 0
    n = 0
    ps = patchsize
    img = burst[n,b]
    print("img",img.shape)

    img_rs = rearrange(img,'c h w -> h w c').cpu()
    patches_v1 = shift_concat_image(img_rs, patchsize)
    [H, W, D, C] = patches_v1.shape
    patches_v1 = patches_v1.reshape([H * W, D * C]).astype(np.float32)
    patches_v1 = torch.tensor(patches_v1)

    
    img_pad = F.pad(img.unsqueeze(0),(ps//2,ps//2,ps//2,ps//2),mode='reflect')
    print("pad",img_pad.shape)
    unfold = nn.Unfold(patchsize,1,0,1)
    patches_v2 = rearrange(unfold(img_pad),'1 l r -> r l')
    
    print("v1",patches_v1.shape)
    print("v2",patches_v2.shape)
    print("l2",F.mse_loss(patches_v1,patches_v2.cpu()))

    patches_v2 = rearrange(patches_v2,'r (c ps1 ps2) -> r ps1 ps2 c',ps1=ps,ps2=ps)
    patches_v2[...,ps//2,ps//2,:] = 0
    patches_v2 = rearrange(patches_v2,'r ps1 ps2 c -> r (ps1 ps2 c)')

    print("l2-zmid",F.mse_loss(patches_v1,patches_v2.cpu()))

    print(patches_v1[0].reshape(3,3,3))
    print(patches_v2[0].reshape(3,3,3))

    
def compute_similar_bursts_n2sim(cfg,burst,K,patchsize=3):
    N,B = burst.shape[:2]
    burst = burst.cpu()
    sim_burst = []
    for b in range(B):
        sim_frames = []
        for n in range(N):
            image = rearrange(burst[n,b],'c h w -> h w c')
            sim_images = compute_sim_images(image, patchsize, K, img_ori=None) 
            sim_images = rearrange(sim_images,'k h w c -> k c h w')
            sim_images = torch.tensor(sim_images)
            sim_frames.append(sim_images)
        sim_frames = torch.stack(sim_frames,dim=0)
        sim_burst.append(sim_frames)
    sim_burst = torch.stack(sim_burst,dim=1).to(cfg.gpuid,non_blocking=True)
    # for k in range(K):
    #     print(k,F.mse_loss(sim_burst[:,:,k],burst))
    return sim_burst

async def compute_kindex_rands_async(cfg,burst,K):
    N,B,C,H,W = burst.shape
    kindex = []
    for b in range(B):
        kindex_b = []
        for n in range(N):
            kindex_b.append(torch.stack([torch.randperm(K+1) for _ in range(C*H*W)]).t().long().to(cfg.gpuid))
        kindex.append(torch.stack(kindex_b))
    kindex = torch.stack(kindex)
    return kindex

async def compute_similar_bursts_async(cfg,burst,K,patchsize=3,shuffle_k=True,kindex=None):
    compute_similar_bursts(cfg,burst,K,patchsize=3,shuffle_k=shuffle_k,kindex=kindex)


def tile_patches(burst,patchsize,search_method):
    """
    prepares a sequence of patches centered at each pixel location
    """
    # -- backward compat --
    pix_only = not isinstance(burst,edict)
    burst = convert_edict(burst)

    # -- init --
    N,B = burst.shape[:2]
    ps = patchsize
    unfold = nn.Unfold(ps,1,0,1)
    patches = edict()

    # -- tile pixel patches --
    pix_pad = rearrange(burst.pix,'n b c h w -> (b n) c h w')
    pix_pad = F.pad(pix_pad,(ps//2,ps//2,ps//2,ps//2),mode='reflect')
    patches.pix = unfold(pix_pad)

    # -- tile feature patches --
    if pix_only:
        patches.pix = rearrange(patches.pix,'bn (c ps1 ps2) r -> bn r ps1 ps2 c',ps1=ps,ps2=ps)
        if search_method == "l2":
            patches.ftr = patches.pix.clone()
            # patches.ftr[...,ps//2,ps//2,:] = 0
        else: patches.ftr = patches.pix
        shape_str = '(b n) r ps1 ps2 c -> b n r (ps1 ps2 c)'
        patches.pix = rearrange(patches.pix,shape_str,n=N)
        patches.ftr = rearrange(patches.ftr,shape_str,n=N)
    else:
        ftr_pad = rearrange(burst.ftr,'n b c h w -> (b n) c h w')
        ftr_pad = F.pad(ftr_pad,(ps//2,ps//2,ps//2,ps//2),mode='reflect')
        patches.ftr = unfold(ftr_pad)
        shape_str = '(b n) (c ps1 ps2) r -> b n r (ps1 ps2 c)'
        patches.pix = rearrange(patches.pix,shape_str,b=B,ps1=ps,ps2=ps)
        patches.ftr = rearrange(patches.ftr,shape_str,b=B,ps1=ps,ps2=ps)

    # -- contiguous for faiss --
    patches.pix = patches.pix.contiguous()
    patches.ftr = patches.ftr.contiguous()
    patches.shape = patches.pix.shape

    # -- return for backward compat --
    if pix_only: return patches.pix,patches.ftr
    else: return patches
    
def compute_similar_bursts(cfg,burst_query,burst_database,K,noise_level,patchsize=3,shuffle_k=True,kindex=None,pick_2=False,only_middle=True,search_method="l2",db_level="burst"):
    """
    params: burst shape: [N, B, C, H, W]
    """
    # -- error checking --
    if not(search_method in ["l2","w"]):
        raise ValueError(f"Uknown search method [{search_method}]")
    if not (db_level in ["batch","burst","frame"]):
        raise ValueError(f"Invalid Reference Database [{db_level}]")

    # -- init shapes --
    ps = patchsize
    N,B,C,H,W = burst_query.shape
    Nd,Bd,Cd,Hd,Wd = burst_database.shape

    # -- tile patches --
    q_patches,q_patches_zc = tile_patches(burst_query,patchsize,search_method)
    B,N,R,ND = q_patches.shape

    db_patches,db_patches_zc = tile_patches(burst_database,patchsize,search_method)
    Bd,Nd,Rd,NDd = db_patches.shape

    # -- faiss setup --
    res = faiss.StandardGpuResources()
    faiss_cfg = faiss.GpuIndexFlatConfig()
    faiss_cfg.useFloat16 = False
    faiss_cfg.device = cfg.gpuid

    # -- search across entire batch --
    if db_level == "batch":
        database = rearrange(db_patches,'b n r l -> (b n r) l')
        database_zc = rearrange(db_patches_zc,'b n r l -> (b n r) l')
        cdatabase = rearrange(clean_patches,'b n r l -> (b n r) l')        
        gpu_index = faiss.GpuIndexFlatL2(res, ND, faiss_cfg)
        gpu_index.add(database_zc)

    sim_images = []
    for b in range(B):

        # -- setup database --
        if db_level == "burst":
            database = rearrange(db_patches[b],'n r l -> (n r) l')
            database_zc = rearrange(db_patches_zc[b],'n r l -> (n r) l')
            gpu_index = faiss.GpuIndexFlatL2(res, ND, faiss_cfg)
            gpu_index.add(database_zc)

        sim_frames = []
        for n in range(N):
            if only_middle and n != N//2: continue

            # -- setup database --
            if db_level == "frame":
                database = db_patches[b,n]
                database_zc = db_patches_zc[b,n]
                gpu_index = faiss.GpuIndexFlatL2(res, ND, faiss_cfg)
                gpu_index.add(database_zc)

            # -- run faiss search --
            query = q_patches_zc[b,n].contiguous()
            if search_method == "l2":
                # -- locations of smallest K patches; no identity so no first column --
                # D, I = gpu_index.search(query,K+1)
                # S,V = D[:,1:], I[:,1:]

                # -- locations of smallest K patches; no identity so no first column --
                D, I = gpu_index.search(query,K)
                S,V = D,I
            elif search_method == "w":
                search_args = [res,noise_level,database_zc,query,K]
                D, I = search_mod_raw_array_pytorch(*search_args)
                S,V = D,I
            elif search_method == "numba":
                search_args = [res,noise_level,database_zc,query,K]
                D, I = search_raw_array_numba(*search_args)
                S,V = D,I
            else: raise ValueError(f"Uknown search method [{search_method}]")

            # -- reshape to extract middle pixel from nearest patches --
            shape_str = 'hw k (ps1 ps2 c) -> hw k ps1 ps2 c'
            sim_patches = rearrange(database[V],shape_str,ps1=ps,ps2=ps)

            # -- create image from nearest pixels --
            sim_pixels = sim_patches[...,ps//2,ps//2,:]
            shape_str = '(h w) k c -> k (c h w)'
            sim_image = rearrange(sim_pixels,shape_str,h=H)

            # -- concat with original image --
            # image_flat = rearrange(burst_query[n,b],'c h w -> 1 (c h w)')
            # if search_method != "w":
            #     sim_image = torch.cat([image_flat,sim_image],dim=0)

            # -- shuffle across each k --
            R = sim_image.shape[0]
            if shuffle_k:
                if kindex is None:
                    kindex_sim = torch.randint(R,(R,C*H*W,)).long().to(cfg.gpuid)
                else: kindex_sim = kindex[b,n][:R] % R
                sim_image.scatter_(0,kindex_sim,sim_image)
                
            # -- add to frames --
            if pick_2:
                rand2 = torch.randperm(R)[:2]
                sim_image = sim_image[rand2]
                
            shape_str = 'k (c h w) -> 1 k c h w'
            sim_image = rearrange(sim_image,shape_str,h=H,w=W)

            sim_frames.append(sim_image)
                

        # -- concat on NumFrames dimension --
        sim_frames = torch.cat(sim_frames,dim=0)
        sim_images.append(sim_frames)

    # -- create batch dimension --
    sim_images = torch.stack(sim_images,dim=1)

    # -- visually inspect --
    vis = False
    if vis:
        # print(burst_query.shape,sim_images.shape)
        N,B = burst_query.shape[:2]
        res = burst_database - sim_images[:,:,0]
        res = rearrange(res,'n b c h w -> (n b) c h w')
        bursts = rearrange(burst_query,'n b c h w -> (n b) c h w')
        sims = rearrange(sim_images,'n b k c h w -> (n k b) c h w')
        tv_utils.save_image(bursts,'sim_search_tgt.png',nrow=N*B,normalize=True)
        tv_utils.save_image(sims,'sim_search_similar_images.png',nrow=N*B,normalize=True)
        tv_utils.save_image(res,'sim_search_res.png',nrow=N*B,normalize=True)
        print("Saved example images!")
        # tv_utils.save_image(sim_images[0,0],'sim_search_similar_locations.png')



    # -- append to original images --
    burst_sim = sim_images
    # burst_sim = torch.cat([burst.unsqueeze(2),sim_images],dim=2)

    # -- compute differences to check --
    # for k in range(K+1):
    #     print(k,F.mse_loss(burst_sim[:,:,k],burst))
    # exit()

    return burst_sim


def create_k_grid(burst,shuffle=False):
    """
    :params burst: shape: [B,N,G,C,H,W]
    """

    # -- method v0 --
    G = burst.shape[2]
    if G == 2:
        i0,i1 = torch.randperm(G)[:2]
        return [i0],[i1]

    # -- method v1 --
    k_ins,k_outs = [],[]
    for i in range(0,G,2):
        if i+1 >= G: break
        k_ins.append(i),k_outs.append(i+1)

    # -- shuffle order among pairs --
    L = len(k_ins)
    k_ins,k_outs = np.array(k_ins),np.array(k_outs)
    order = npr.permutation(L)
    k_ins,k_outs = k_ins[order],k_outs[order]
    swap = np.array([npr.permutation(2) for _ in range(L)])

    # -- shuffle order within pairs -- (just torch.scatter for numpy)
    kindex = np.c_[k_ins,k_outs]
    for l in range(L):
        k_ins[l],k_outs[l] = kindex[l,swap[l][0]],kindex[l,swap[l][1]]

    return k_ins,k_outs

def create_k_grid_v2(K,shuffle=False,L=None):
    x,y = np.arange(K),np.arange(K)
    xv,yv = np.meshgrid(x,y)
    k_ins,k_outs = xv.ravel(),yv.ravel()

    # -- only keep unique, non-eq pairs --
    pairs = []
    for kin,kout in zip(k_ins,k_outs):
        if kin >= kout: continue
        pairs.append([kin,kout])
    pairs = np.array(pairs)
    P = len(pairs)
    k_ins,k_outs = pairs[:,0],pairs[:,1]

    # -- shuffle if needed --
    if shuffle:
        ordering = npr.permutation(P)
        k_ins,k_outs = k_ins[ordering],k_outs[ordering]

    # -- limit number of combos --
    if not (L is None):
        k_ins,k_outs = k_ins[:L],k_outs[:L]

    return k_ins,k_outs



class kIndexPermLMDB():

    def __init__(self,batch_size,ds_num_frames,img_shape=(3,128,128)):

        # -- init info --
        B = batch_size
        C,H,W = img_shape
        self.batch_size = batch_size
        self.ds_num_frames = ds_num_frames

        # -- lmdb path --
        self.lmdb_path = Path("data/rand_perms/lmdbs/randperm_c3_hw128_nf12_ns8_e1_d10000/")

        # -- extract metadata --
        metadata_fn = self.lmdb_path / Path("metadata.pkl")
        self.meta_info = pickle.load(open(metadata_fn,'rb'))
        self.num_samples = self.meta_info['num_samples']
        self.num_sim = self.meta_info['num_sim']
        self.num_frames = self.meta_info['num_frames']

        # -- create shape --
        N,K = self.num_frames,self.num_sim
        self.shape = (N,K+1,C*H*W)

        # -- init data env --
        self.data_env = None

        # -- randomize ordering --
        self.order = None
        self.shuffle()
        
    def __len__(self):
        return self.meta_info['num_samples']

    def shuffle(self):
        self.order = torch.randperm(self.meta_info['num_samples'])

    def __getitem__(self,batch_index):
        rands = []
        lmdb_indices = self._lmdb_indices_from_batch_idx(batch_index)
        for lmdb_index in lmdb_indices:
            rands.append(self._load_single_lmdb_entry(lmdb_index))
        rands = torch.stack(rands)
        return rands

    def _lmdb_indices_from_batch_idx(self,batch_index):
        nbatches = self.__len__() // self.batch_size
        batch_index = batch_index % nbatches
        lmdb_indices = np.arange(self.batch_size)
        lmdb_indices += batch_index*self.batch_size
        return lmdb_indices

    def _load_single_lmdb_entry(self,lmdb_index,dtype=np.uint8):
        # -- create data env --
        if self.data_env is None:
            self.data_env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
        data_env = self.data_env

        # -- read from lmdb --
        key = "{}_perm".format(self.order[lmdb_index]).encode("ascii")
        with data_env.begin(write=False) as txn: buf = txn.get(key)

        # -- convert to ndarrays --
        indices = np.frombuffer(buf, dtype=dtype).copy().reshape(self.shape)
        indices = torch.tensor(indices).long()

        # -- limit to number of frames actually used --
        indices = indices[:self.ds_num_frames]

        return indices

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#                Burst Analysis
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        
def convert_edict(pix_data):
    if not isinstance(pix_data,edict):
        tmp = pix_data
        pix_data = edict()
        pix_data.pix = tmp
        pix_data.ftr = tmp
        pix_data.shape = tmp.shape
    return pix_data

def compute_similar_bursts_analysis(cfg,burst_query,burst_database,clean_database,K,noise_level,patchsize=3,shuffle_k=True,kindex=None,pick_2=False,only_middle=True,search_method="l2",db_level="burst"):
    """
    params: burst shape: [N, B, C, H, W]
    """
    # -- error checking --
    if not(search_method in ["l2","w","numba"]):
        raise ValueError(f"Uknown search method [{search_method}]")
    if not (db_level in ["batch","burst","frame"]):
        raise ValueError(f"Invalid Reference Database [{db_level}]")

    # -- enable pixels and features --
    burst_query = convert_edict(burst_query)
    burst_database = convert_edict(burst_database)
    clean_database = convert_edict(clean_database)

    # -- init shapes --
    ps = patchsize
    N,B,C,H,W = burst_query.shape
    Nd,Bd,Cd,Hd,Wd = burst_database.shape

    # -- tile patches --
    q_patches = tile_patches(burst_query,patchsize,search_method)
    B,N,R,ND = q_patches.shape

    db_patches = tile_patches(burst_database,patchsize,search_method)
    Bd,Nd,Rd,NDd = db_patches.shape

    clean_patches = tile_patches(clean_database,patchsize,search_method)
    Bd,Nd,Rd,NDd = clean_patches.shape

    ND = q_patches.ftr.shape[-1]

    # -- faiss setup --
    res = faiss.StandardGpuResources()
    faiss_cfg = faiss.GpuIndexFlatConfig()
    faiss_cfg.useFloat16 = False
    faiss_cfg.device = cfg.gpuid

    # -- init searching vars --
    database = edict()
    cdatabase = edict()

    # -- search across entire batch --
    if db_level == "batch":
        database.pix = rearrange(db_patches.pix,'b n r l -> (b n r) l')
        database.ftr = rearrange(db_patches.ftr,'b n r l -> (b n r) l')
        cdatabase.pix = rearrange(clean_patches.pix,'b n r l -> (b n r) l')        
        gpu_index = faiss.GpuIndexFlatL2(res, ND, faiss_cfg)
        gpu_index.add(database.ftr)

    sim_images = []
    csim_images = []
    wsim_images = []
    batch_distances = []
    batch_indices = []
    for b in range(B):

        # -- setup database --
        if db_level == "burst":
            database.pix = rearrange(db_patches.pix[b],'n r l -> (n r) l')
            database.ftr = rearrange(db_patches.ftr[b],'n r l -> (n r) l')
            cdatabase.pix = rearrange(clean_patches.pix[b],'n r l -> (n r) l')
            gpu_index = faiss.GpuIndexFlatL2(res, ND, faiss_cfg)
            gpu_index.add(database.ftr)

        sim_frames = []
        csim_frames = []
        wsim_frames = []
        frame_distances = []
        frame_indices = []
        for n in range(N):
            if only_middle and n != N//2: continue

            # -- setup database --
            if db_level == "frame":
                database.pix = db_patches.pix[b,n]
                database.ftr = db_patches.ftr[b,n]
                cdatabase.pix = clean_patches.pix[b,n]
                gpu_index = faiss.GpuIndexFlatL2(res, ND, faiss_cfg)
                gpu_index.add(database.ftr)

            # -- run faiss search --
            query_ftr = q_patches.ftr[b,n].contiguous()
            if search_method == "l2":
                D, I = gpu_index.search(query_ftr,K)
                # -- locations of smallest K patches; no identity so no first column --
                S,V = D[:,:], I[:,:]
            elif search_method == "w":
                search_args = [res,noise_level,database.ftr,query_ftr,K]
                D, I = search_mod_raw_array_pytorch(*search_args)
                S,V = D,I
            elif search_method == "numba":
                search_args = [res,noise_level,database.ftr,query_ftr,K]
                D, I = search_raw_array_numba(*search_args)
                S,V = D,I
            else: raise ValueError(f"Uknown search method [{search_method}]")

            # -- add information to distances and indices --
            frame_distances.append(S)
            frame_indices.append(V)

            # -- extract middle pixel from patch in clean image --
            shape_str = 'hw k (ps1 ps2 c) -> hw k ps1 ps2 c'
            csim_patches = rearrange(cdatabase.pix[V],shape_str,ps1=ps,ps2=ps)
            csim_pixels = csim_patches[...,ps//2,ps//2,:]
            shape_str = '(h w) k c -> 1 k c h w'
            csim_image = rearrange(csim_pixels,shape_str,h=H)

            # -- reshape to extract middle pixel from nearest patches --
            shape_str = 'hw k (ps1 ps2 c) -> hw k ps1 ps2 c'
            sim_patches = rearrange(database.pix[V],shape_str,ps1=ps,ps2=ps)

            # -- create image from nearest pixels --
            sim_pixels = sim_patches[...,ps//2,ps//2,:]
            shape_str = '(h w) k c -> k (c h w)'
            sim_image = rearrange(sim_pixels,shape_str,h=H)

            # -- concat with original image --
            # image_flat = rearrange(burst_query[n,b],'c h w -> 1 (c h w)')
            # if search_method != "w":
            #     sim_image = torch.cat([image_flat,sim_image],dim=0)

            # -- shuffle across each k --
            R = sim_image.shape[0]
            if shuffle_k:
                if kindex is None:
                    kindex_sim = torch.randint(R,(R,C*H*W,)).long().to(cfg.gpuid)
                else:
                    kindex_sim = kindex[b,n]
                sim_image.scatter_(0,kindex_sim,sim_image)
                
            # -- add to frames --
            if pick_2:
                rand2 = torch.randperm(R)[:2]
                sim_image = sim_image[rand2]
                
            shape_str = 'k (c h w) -> 1 k c h w'
            sim_image = rearrange(sim_image,shape_str,h=H,w=W)

            sim_frames.append(sim_image)
            csim_frames.append(csim_image)
                
        # -- extra info --
        shape_str = 'n (h w) k1 -> n k1 h w'
        frame_distances = rearrange(torch.stack(frame_distances,dim=0),shape_str,h=H)
        frame_indices = rearrange(torch.stack(frame_indices,dim=0),shape_str,h=H)
        batch_distances.append(frame_distances)
        batch_indices.append(frame_indices)

        # -- clean concat --
        csim_frames = torch.cat(csim_frames,dim=0)
        csim_images.append(csim_frames)

        # -- concat on NumFrames dimension --
        sim_frames = torch.cat(sim_frames,dim=0)
        sim_images.append(sim_frames)

    # -- create batch dimension --
    batch_distances = torch.stack(batch_distances,dim=1)    
    batch_indices = torch.stack(batch_indices,dim=1)    
    csim_images = torch.stack(csim_images,dim=1)
    sim_images = torch.stack(sim_images,dim=1)

    # -- visually inspect --
    vis = False
    if vis:
        sims = rearrange(sim_images[:,0],'n k c h w -> (n k) c h w')
        tv_utils.save_image(burst_query[:,0],'sim_search_tgt.png',nrow=N,normalize=True)
        tv_utils.save_image(sims,'sim_search_similar_images.png',nrow=K,normalize=True)
        print("Saved example images!")

    return sim_images,csim_images,wsim_images,batch_distances,batch_indices



