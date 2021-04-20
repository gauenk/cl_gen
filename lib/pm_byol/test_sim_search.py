
# -- python imports --
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tvT
from torchvision.transforms import functional as tvF
from torchvision import utils as tv_utils

# -- project imports --
from datasets.transforms import get_noise_transform
from n2sim.sim_search import compute_similar_bursts_analysis
from pyutils.misc import images_to_psnrs

# -- [local] project imports --
from n2sim.debug import print_tensor_stats
from .noise_settings import create_noise_level_grid,get_keys_noise_level_grid
from .utils import print_tensor_stats

def test_sim_search_serial_batch(cfg,clean,model):
    B = clean.shape[1]
    keys = get_keys_noise_level_grid(cfg)
    results = edict()
    results.ftr = {}
    results.pix = {}
    for b in range(0,B):
        # -- compute psnr --
        psnrs_sim = test_sim_search(cfg,clean[:,[b]],model)
        psnrs_ftr = psnrs_sim[cfg.byol_backbone_name]
        psnrs_pix = psnrs_sim["pix"]
        for name in psnrs_ftr.keys():
            if name not in results.ftr.keys():
                results.ftr[name] = edict({})
                results.ftr[name].psnrs = []
            if name not in results.pix.keys():
                results.pix[name] = edict({})
                results.pix[name].psnrs = []
            results.ftr[name].psnrs.extend(psnrs_ftr[name].psnrs)
            results.pix[name].psnrs.extend(psnrs_pix[name].psnrs)
    for key in keys:
        compute_psnrs_summary(results.ftr[key])
        compute_psnrs_summary(results.pix[key])
    results[cfg.byol_backbone_name] = results.ftr
    return results

def test_sim_search(cfg,clean,model):
    if cfg.byol_backbone_name == "unet":
        return test_sim_search_ftr(cfg,clean,model,["unet","pix"])
    elif cfg.byol_backbone_name == "attn":
        return test_sim_search_ftr(cfg,clean,model,["attn","pix"])
    else: raise ValueError("Uknown backbone for byol [{cfg.byol_backbone_name}]")

def test_sim_search_unet(cfg,clean,model):
    test_sim_search_ftr(cfg,clean,model,["unet"])

def test_sim_search_attn(cfg,clean,model):
    test_sim_search_ftr(cfg,clean,model,["attn"])

def test_sim_search_pix(cfg,clean,model):
    test_sim_search_ftr(cfg,clean,model,["pix"])

def get_feature_image(cfg,noisy_patches,model,ftr_type):
    if ftr_type == "pix":
        return get_pixel_features(cfg,noisy_patches)
    elif ftr_type == "unet":
        return get_unet_features(cfg,noisy_patches,model)
    elif ftr_type == "attn":
        return get_attn_features(cfg,noisy_patches,model)
    else: raise ValueError(f"Uknown feature type [{ftr_type}]")

def get_pixel_features(cfg,noisy_patches):
    f_mid = cfg.byol_nh_size**2//2
    p_mid = cfg.byol_patchsize//2
    noisy_img = noisy_patches[:,:,:,f_mid,:,p_mid,p_mid]
    noisy_img = rearrange(noisy_img,'(h w) n b c -> n b c h w',h=cfg.frame_size)
    return noisy_img

def get_attn_features(cfg,noisy_patches,model):
    # -- get noisy features a --
    noisy_a = rearrange(noisy_patches,'r n b l c h w -> n (r b) l c h w')
    # noisy_a = rearrange(noisy_a,'n b l c h w -> (n b l) c h w') # "form_input_patches" replaces this
    noisy_inputs = model.patch_helper.form_input_patches(noisy_a)
    embeddings_0 = model(noisy_inputs,return_embedding=True)
    ftr_img_0 = model.patch_helper.embeddings_to_image(embeddings_0)

    # -- get noisy features b --
    noisy_b = torch.flip(noisy_patches,dims=(1,)) # reverse
    # noisy_b = noisy_patches
    noisy_b = rearrange(noisy_b,'r n b l c h w -> n (r b) l c h w')
    # noisy_b = rearrange(noisy_b,'n b l c h w -> (n b l) c h w') # "form_input_patches" replaces this
    noisy_inputs = model.patch_helper.form_input_patches(noisy_b)
    embeddings_1 = model(noisy_b,return_embedding=True)
    ftr_img_1 = model.patch_helper.embeddings_to_image(embeddings_1)

    # -- stack ftr images --
    ftr_img = torch.cat([ftr_img_0,ftr_img_1],dim=0)
    return ftr_img

def get_unet_features(cfg,noisy_patches,model):

    # -- get noisy features a --
    noisy_a = rearrange(noisy_patches,'r n b l c h w -> n (r b) l c h w')
    noisy_inputs = model.patch_helper.form_input_patches(noisy_a)
    embeddings_0 = model(noisy_inputs,return_embedding=True)
    # embeddings_0 = F.normalize(embeddings_0, dim=-1, p=2)
    ftr_img_0 = model.patch_helper.embeddings_to_image(embeddings_0)

    # -- get noisy features b --
    noisy_b = torch.flip(noisy_patches,dims=(1,))
    noisy_b = rearrange(noisy_b,'r n b l c h w -> n (r b) l c h w')
    noisy_inputs = model.patch_helper.form_input_patches(noisy_b)
    embeddings_1 = model(noisy_inputs,return_embedding=True)
    # embeddings_1 = F.normalize(embeddings_1, dim=-1, p=2)
    ftr_img_1 = model.patch_helper.embeddings_to_image(embeddings_1)

    # -- stack ftr images --
    ftr_img = torch.cat([ftr_img_0,ftr_img_1],dim=0)
    return ftr_img


def test_sim_search_ftr(cfg,clean,model,ftr_types):

    # -- init --
    N,B,C,H,W = clean.shape
    ps = cfg.byol_patchsize
    if clean.min() < 0: clean += 0.5 # non-negative pixels

    # -- unfold clean image --
    patches = model.patch_helper.prepare_burst_patches(clean)
    patches = patches.cuda(non_blocking=True)
    ps = cfg.byol_patchsize

    # shape = (r n b nh_size^2 c ps_B ps_B)

    # -- start loop --
    psnrs = edict({})
    for ftr_type in ftr_types: psnrs[ftr_type] = edict({})
    noisy_grid = create_noise_level_grid(cfg)
    with torch.no_grad():
        for noise_params in noisy_grid:
            
            # -- setup noise xform --
            cfg.noise_type = noise_params.ntype
            cfg.noise_params.ntype = cfg.noise_type
            cfg.noise_params[cfg.noise_type] = noise_params
            noise_func = get_noise_transform(cfg.noise_params,noise_only=True)
    
            # -- apply noise --
            noisy_patches = noise_func(patches) # shape = (r n b nh_size^2 c ps_B ps_B)
    
            # -- create noisy img --
            noisy_img = get_pixel_features(cfg,noisy_patches)
    
            # -- get features --
            for ftype in ftr_types:
                ftr_img = get_feature_image(cfg,noisy_patches,model,ftype)
            
                # -- some debugging code --
                vis = False
                if vis: vis_noisy_features(cfg,noisy_img,ftr_img,clean,ftype)

                testing_indexing = False
                if testing_indexing:
                    test_patch_helper_indexing(cfg,noisy_img,ftr_img,clean,ftype)

                # -- construct similar image --
                if ftype != "pix":
                    sim_patchsize = cfg.sim_patchsize
                    cfg.sim_patchsize = 1
                    psnrs_np = compute_similar_psnr(cfg,noisy_img,ftr_img,clean)
                    cfg.sim_patchsize = sim_patchsize
                else:
                    psnrs_np = compute_similar_psnr(cfg,noisy_img,ftr_img,clean)
                    
                # -- compute psnr --
                psnrs[ftype][noise_params.name] = edict()
                psnrs[ftype][noise_params.name].psnrs = psnrs_np
                compute_psnrs_summary(psnrs[ftype][noise_params.name])
                # psnrs[ftype][noise_params.name].ave = np.mean(psnrs_np)
                # psnrs[ftype][noise_params.name].std = np.std(psnrs_np)
                # psnrs[ftype][noise_params.name].min = np.min(psnrs_np)
                # psnrs[ftype][noise_params.name].max = np.max(psnrs_np)
                del ftr_img
    
    return psnrs

def compute_psnrs_summary(psnr_df):
    psnrs_np = psnr_df.psnrs
    psnr_df.ave = np.mean(psnrs_np)
    psnr_df.std = np.std(psnrs_np)
    psnr_df.min = np.min(psnrs_np)
    psnr_df.max = np.max(psnrs_np)

def compute_similar_psnr(cfg,noisy_img,ftr_img,clean,crop=False):

    # -- construct similar image --
    query = edict()
    query.pix = noisy_img[[0]]
    query.ftr = ftr_img[[0]]
    query.shape = query.pix.shape

    database = edict()
    database.pix = noisy_img[[1]]
    database.ftr = ftr_img[[1]]
    database.shape = database.pix.shape

    clean_db = edict()
    clean_db.pix = clean[[1]]
    clean_db.ftr = clean_db.pix
    clean_db.shape = clean_db.pix.shape

    sim_outputs = compute_similar_bursts_analysis(cfg,query,database,clean_db,1,
                                                  patchsize=cfg.sim_patchsize,
                                                  shuffle_k=False,kindex=None,
                                                  only_middle=cfg.sim_only_middle,
                                                  db_level='frame',
                                                  search_method=cfg.sim_method,
                                                  noise_level=None)

    # -- compute psnr --
    ref = clean[0]
    clean_sims = sim_outputs[1][0,:,0]
    if crop:
        ref = tvF.crop(ref,10,10,48,48)
        clean_sims = tvF.crop(clean_sims,10,10,48,48)
    psnrs_np = images_to_psnrs(ref.cpu(),clean_sims.cpu())
    return psnrs_np

def vis_noisy_features(cfg,noisy_img,ftr_img,clean,ftype):
    if cfg.noise_type == "none" and ftype != "pix":
        N,B = ftr_img.shape[:2]
        print(N,B,ftr_img.shape,clean.shape)
        print_tensor_stats("noisy",noisy_img)
        print_tensor_stats("ftr_img",ftr_img[:,:,:3])
        save_noisy_image = rearrange(noisy_img,'n b c h w -> (n b) c h w')
        tv_utils.save_image(save_noisy_image,"noisy.png",normalize=True)
        save_ftr_image = rearrange(ftr_img,'n b c h w -> (n b) c h w')
        tv_utils.save_image(save_ftr_image[:,:3],"ftr_image_3.png",normalize=True)
        tv_utils.save_image(save_ftr_image[:,3:6],"ftr_image_6.png",normalize=True)
    
def test_patch_helper_indexing(cfg,noisy_img,ftr_img,clean,ftype):
    if ftype != "pix":
        print_tensor_stats("clean",clean)
        print(noisy_img.shape,ftr_img.shape)
        fmse1 = F.mse_loss(noisy_img[[0]],ftr_img[[0],:,:3])
        fmse2 = F.mse_loss(noisy_img[[1]],ftr_img[[1],:,:3])
        print( "[fmse]: %2.2e and %2.2e" % ( fmse1, fmse2 ) )

        fmse1 = F.mse_loss(clean[[0]],ftr_img[[0],:,:3])
        fmse2 = F.mse_loss(clean[[1]],ftr_img[[1],:,:3])
        print( "[fmse-c]: %2.2e and %2.2e" % ( fmse1, fmse2 ) )



#
# DEPRECATED
# 

def test_sim_search_pix_v2(cfg,clean,model):

    # -- init --
    N,B,C,H,W = clean.shape
    cleanBN = rearrange(clean,'n b c h w -> (b n) c h w')
    clean_pil = [tvT.ToPILImage()(cleanBN[i]+0.5).convert("RGB") for i in range(B*N)]
    ps = cfg.byol_patchsize
    unfold = nn.Unfold(ps,1,0,1)

    # -- start loop --
    psnrs = {}
    noisy_grid = create_noise_level_grid(cfg)
    for noise_params in noisy_grid:
        
        # -- get noisy images --
        cfg.noise_type = noise_params.ntype
        cfg.noise_params.ntype = cfg.noise_type
        cfg.noise_params[cfg.noise_type] = noise_params
        noise_func = get_noise_transform(cfg.noise_params)
        noisyBN = torch.stack([noise_func(clean_pil[i]) for i in range(B*N)],dim=0)
        noisy = rearrange(noisyBN,'(b n) c h w -> n b c h w',b=B)

        # -- construct similar image --
        query = edict()
        query.pix = noisy[[0]]
        query.ftr = noisy[[0]]
        query.shape = query.pix.shape

        database = edict()
        database.pix = noisy[[1]]
        database.ftr = noisy[[1]]
        database.shape = database.pix.shape

        clean_db = edict()
        clean_db.pix = clean[[1]]
        clean_db.ftr = clean_db.pix
        clean_db.shape = clean_db.pix.shape

        sim_outputs = compute_similar_bursts_analysis(cfg,query,database,clean_db,1,
                                                      patchsize=cfg.sim_patchsize,
                                                      shuffle_k=False,kindex=None,
                                                      only_middle=cfg.sim_only_middle,
                                                      db_level='frame',
                                                      search_method=cfg.sim_method,
                                                      noise_level=None)

        # -- compute psnr --
        ref = clean[0]
        clean_sims = sim_outputs[1][0,:,0]
        psnrs_np = images_to_psnrs(ref.cpu(),clean_sims.cpu())
        psnrs[noise_params.name] = edict()
        psnrs[noise_params.name].psnrs = psnrs_np
        psnrs[noise_params.name].ave = np.mean(psnrs_np)
        psnrs[noise_params.name].std = np.std(psnrs_np)
        psnrs[noise_params.name].min = np.min(psnrs_np)
        psnrs[noise_params.name].max = np.max(psnrs_np)
        # print(noise_params.name,psnrs[noise_params.name])

    return psnrs


def test_sim_search_unet_v2(cfg,clean,model):

    # -- init --
    N,B,C,H,W = clean.shape
    ps = cfg.byol_patchsize

    # -- unfold clean image --
    patches = model.patch_helper.prepare_burst_patches(clean)
    patches = patches.cuda(non_blocking=True)
    # R,N,B,L,C,H,W = patches.shape
    
    # -- start loop --
    psnrs = {}
    noisy_grid = create_noise_level_grid(cfg)
    for noise_params in noisy_grid:
        
        # -- setup noise xform --
        cfg.noise_type = noise_params.ntype
        cfg.noise_params.ntype = cfg.noise_type
        cfg.noise_params[cfg.noise_type] = noise_params
        noise_func = get_noise_transform(cfg.noise_params,use_to_tensor=False)

        # -- apply noise --
        noisy_patches = noise_func(patches) # shape = (r n b nh_size^2 c ps_B ps_B)

        # -- create noisy img --
        f_mid = cfg.byol_nh_size**2//2
        p_mid = cfg.byol_patchsize//2
        noisy_img = noisy_patches[:,:,:,f_mid,:,p_mid,p_mid]
        noisy_img = rearrange(noisy_img,'(h w) n b c -> n b c h w',h=cfg.frame_size)

        ftr_img = get_feature_image(cfg,noisy_patches,model,"unet")

        print("[ftr_img.shape]",ftr_img.shape)
        # print("[emd] PSNR: ",np.mean(images_to_psnrs(embeddings_0,embeddings_1)))
        # print("[ftr] PSNR: ",np.mean(images_to_psnrs(ftr_img_0,ftr_img_1)))

        # -- construct similar image --
        query = edict()
        query.pix = noisy_img[[0]]
        query.ftr = ftr_img[[0]]
        query.shape = query.pix.shape

        database = edict()
        database.pix = noisy_img[[1]]
        database.ftr = ftr_img[[1]]
        database.shape = database.pix.shape

        clean_db = edict()
        clean_db.pix = clean[[1]]
        clean_db.ftr = clean_db.pix
        clean_db.shape = clean_db.pix.shape

        sim_outputs = compute_similar_bursts_analysis(cfg,query,database,clean_db,1,
                                                      patchsize=cfg.sim_patchsize,
                                                      shuffle_k=False,kindex=None,
                                                      only_middle=cfg.sim_only_middle,
                                                      db_level='frame',
                                                      search_method=cfg.sim_method,
                                                      noise_level=None)

        # -- compute psnr --
        ref = clean[0]
        clean_sims = sim_outputs[1][0,:,0]
        psnrs_np = images_to_psnrs(ref.cpu(),clean_sims.cpu())
        psnrs[noise_params.name] = edict()
        psnrs[noise_params.name].psnrs = psnrs_np
        psnrs[noise_params.name].ave = np.mean(psnrs_np)
        psnrs[noise_params.name].std = np.std(psnrs_np)
        psnrs[noise_params.name].min = np.min(psnrs_np)
        psnrs[noise_params.name].max = np.max(psnrs_np)
        # print(noise_params.name,psnrs[noise_params.name])

    return psnrs


def test_sim_search_attn_v2(cfg,clean,model):

    # -- init --
    N,B,C,H,W = clean.shape
    ps = cfg.byol_patchsize

    # -- unfold clean image --
    patches = model.patch_helper.prepare_burst_patches(clean)
    patches = patches.cuda(non_blocking=True)
    # R,N,B,L,C,H,W = patches.shape
    
    # -- start loop --
    psnrs = {}
    noisy_grid = create_noise_level_grid(cfg)
    for noise_params in noisy_grid:
        
        # -- setup noise xform --
        cfg.noise_type = noise_params.ntype
        cfg.noise_params.ntype = cfg.noise_type
        cfg.noise_params[cfg.noise_type] = noise_params
        noise_func = get_noise_transform(cfg.noise_params,use_to_tensor=False)

        # -- apply noise --
        noisy_patches = noise_func(patches) # shape = (r n b nh_size^2 c ps_B ps_B)

        # -- create noisy img --
        f_mid = cfg.byol_nh_size**2//2
        p_mid = cfg.byol_patchsize//2
        noisy_img = noisy_patches[:,:,:,f_mid,:,p_mid,p_mid]
        noisy_img = rearrange(noisy_img,'(h w) n b c -> n b c h w',h=cfg.frame_size)

        ftr_img = get_feature_image(cfg,noisy_patches,model,"attn")

        print("[ftr_img.shape]",ftr_img.shape)
        # print("[emd] PSNR: ",np.mean(images_to_psnrs(embeddings_0,embeddings_1)))
        # print("[ftr] PSNR: ",np.mean(images_to_psnrs(ftr_img_0,ftr_img_1)))

        # -- construct similar image --
        query = edict()
        query.pix = noisy_img[[0]]
        query.ftr = ftr_img[[0]]
        query.shape = query.pix.shape

        database = edict()
        database.pix = noisy_img[[1]]
        database.ftr = ftr_img[[1]]
        database.shape = database.pix.shape

        clean_db = edict()
        clean_db.pix = clean[[1]]
        clean_db.ftr = clean_db.pix
        clean_db.shape = clean_db.pix.shape

        sim_outputs = compute_similar_bursts_analysis(cfg,query,database,clean_db,1,
                                                      patchsize=cfg.sim_patchsize,
                                                      shuffle_k=False,kindex=None,
                                                      only_middle=cfg.sim_only_middle,
                                                      db_level='frame',
                                                      search_method=cfg.sim_method,
                                                      noise_level=None)

        # -- compute psnr --
        ref = clean[0]
        clean_sims = sim_outputs[1][0,:,0]
        psnrs_np = images_to_psnrs(ref.cpu(),clean_sims.cpu())
        psnrs[noise_params.name] = edict()
        psnrs[noise_params.name].psnrs = psnrs_np
        psnrs[noise_params.name].ave = np.mean(psnrs_np)
        psnrs[noise_params.name].std = np.std(psnrs_np)
        psnrs[noise_params.name].min = np.min(psnrs_np)
        psnrs[noise_params.name].max = np.max(psnrs_np)
        # print(noise_params.name,psnrs[noise_params.name])

    return psnrs


def print_psnr_results(psnrs,title):
    write_info = (title,psnrs['clean'].ave,psnrs['clean'].std,
                  psnrs['clean'].min,psnrs['clean'].max)
    title = "%s" % title
    sep = "|"

    name = 'clean'
    name_str = "[c]:"
    ave_std = "%2.2f +/- %2.2f" % (psnrs[name].ave,psnrs[name].std)
    min_max = "(%2.2f,%2.2f)" % (psnrs[name].min,psnrs[name].max)
    w_str = f"{title: >15} {name_str: <10} {ave_std: >18} {sep: ^4} {min_max: >12}"
    print(w_str)

    title = ""
    name = 'g-25p0'
    name_str = "[g-25]:"
    ave_std = "%2.2f +/- %2.2f" % (psnrs[name].ave,psnrs[name].std)
    min_max = "(%2.2f,%2.2f)" % (psnrs[name].min,psnrs[name].max)
    w_str = f"{title: >15} {name_str: <10} {ave_std: >18} {sep: ^4} {min_max: >12}"
    print(w_str)

    title = ""
    name = 'g-75p0'
    name_str = "[g-75]:"
    ave_std = "%2.2f +/- %2.2f" % (psnrs[name].ave,psnrs[name].std)
    min_max = "(%2.2f,%2.2f)" % (psnrs[name].min,psnrs[name].max)
    w_str = f"{title: >15} {name_str: <10} {ave_std: >18} {sep: ^4} {min_max: >12}"
    print(w_str)

    title =""
    name = 'pn-4p0-0p0'
    name_str = "[pn-4]:"
    ave_std = "%2.2f +/- %2.2f" % (psnrs[name].ave,psnrs[name].std)
    min_max = "(%2.2f,%2.2f)" % (psnrs[name].min,psnrs[name].max)
    w_str = f"{title: >15} {name_str: <10} {ave_std: >18} {sep: ^4} {min_max: >12}"
    print(w_str)
    
    
