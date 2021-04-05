
# -- python imports --
import bm3d
import asyncio
import pandas as pd
import numpy as np
import numpy.random as npr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from einops import rearrange, repeat, reduce
from easydict import EasyDict as edict

# -- just to draw an fing arrow --
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# -- faiss imports --
import faiss
import faiss.contrib.torch_utils

# -- pytorch imports --
import torch
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms as tvT
import torchvision.transforms.functional as tvF

# -- project code --
import settings
from pyutils.timer import Timer
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr,images_to_psnrs
from datasets.transforms import ScaleZeroMean,RandomChoice
from layers.ot_pytorch import sink_stabilized,sink,pairwise_distances
from layers.burst import BurstRecLoss,EntropyLoss
from datasets.transforms import get_noise_transform

# -- [local] project code --
from n2nwl.dist_loss import ot_pairwise_bp,ot_gaussian_bp,ot_pairwise2gaussian_bp,kl_gaussian_bp,w_gaussian_bp,kl_gaussian_bp_patches
from n2nwl.misc import AlignmentFilterHooks
from n2nwl.plot import plot_histogram_residuals_batch,plot_histogram_gradients,plot_histogram_gradient_norms
from n2sim.sim_search import compute_similar_bursts,compute_similar_bursts_async,compute_similar_bursts_n2sim,create_k_grid,compare_sim_patches_methods,compare_sim_images_methods,compute_kindex_rands_async,kIndexPermLMDB
from n2sim.debug import print_tensor_stats
from .test_sim_search import test_sim_search,test_sim_search_pix,print_psnr_results,test_sim_search_serial_batch
from .noise_settings import get_keys_noise_level_grid
from .test_ps_nh_sizes import test_ps_nh_sizes

def print_tensor_stats(prefix,tensor):
    stats_fmt = (tensor.min().item(),tensor.max().item(),tensor.mean().item())
    stats_str = "%2.2e,%2.2e,%2.2e" % stats_fmt
    print(prefix,stats_str)

async def say_after(delay, what):
    await asyncio.sleep(delay)
    print("Hello World")

async def run_both(cfg,burst,K,patchsize,kindex):
    both = asyncio.gather(
        compute_kindex_rands_async(cfg,burst,K),
        compute_similar_bursts_async(cfg,burst,K,patchsize=patchsize,kindex=kindex)
    )
    await both
    return both

def sample_burst_patches(cfg,model,burst):
    n_indices = 2
    indices = np.random.choice(cfg.frame_size**2,n_indices)
    patches = []
    for index in indices:
        index_window = model.patch_helper.index_window(index,ps=3)
        for nh_index in index_window:
            patches_i = model.patch_helper.gather_local_patches(burst, nh_index)
            patches.append(patches_i)
    patches = torch.cat(patches,dim=1)
    return patches

def train_loop(cfg,model,optimizer,scheduler,train_loader,epoch,record_losses,writer):

    # -=-=-=-=-=-=-=-=-=-=-
    #
    #    Setup for epoch
    #
    # -=-=-=-=-=-=-=-=-=-=-

    model.train()
    model = model.to(cfg.gpuid)

    N = cfg.N
    total_loss = 0
    running_loss = 0
    szm = ScaleZeroMean()
    blocksize = 128
    unfold = torch.nn.Unfold(blocksize,1,0,blocksize)
    use_record = False
    if record_losses is None: record_losses = pd.DataFrame({'burst':[],'ave':[],'ot':[],'psnr':[],'psnr_std':[]})
    noise_type = cfg.noise_params.ntype

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Init Record Keeping
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    align_mse_losses,align_mse_count = 0,0
    rec_mse_losses,rec_mse_count = 0,0
    rec_ot_losses,rec_ot_count = 0,0
    running_loss,total_loss = 0,0

    write_examples = False
    write_examples_iter = 200
    noise_level = cfg.noise_params['g']['stddev']

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #   Load Pre-Simulated Random Numbers
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
    if cfg.use_kindex_lmdb: kindex_ds = kIndexPermLMDB(cfg.batch_size,cfg.N)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Dataset Augmentation
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    transforms = [tvF.vflip,tvF.hflip,tvF.rotate]
    aug = RandomChoice(transforms)
    def apply_transformations(burst,gt_img):
        N,B = burst.shape[:2]
        gt_img_rs = rearrange(gt_img,'b c h w -> 1 b c h w')
        all_images = torch.cat([gt_img_rs,burst],dim=0)
        all_images = rearrange(all_images,'n b c h w -> (n b) c h w')
        tv_utils.save_image(all_images,'aug_original.png',nrow=N+1,normalize=True)
        aug_images = aug(all_images)
        tv_utils.save_image(aug_images,'aug_augmented.png',nrow=N+1,normalize=True)
        aug_images = rearrange(aug_images,'(n b) c h w -> n b c h w',b=B)
        aug_gt_img = aug_images[0]
        aug_burst = aug_images[1:]
        return aug_burst,aug_gt_img

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Half Precision
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # model.align_info.model.half()
    # model.denoiser_info.model.half()
    # model.unet_info.model.half()
    # models = [model.align_info.model,
    #           model.denoiser_info.model,
    #           model.unet_info.model]
    # for model_l in models:
    #     model_l.half()
    #     for layer in model_l.modules():
    #         if isinstance(layer, torch.nn.BatchNorm2d):
    #             layer.float()

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Init Loss Functions
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    alignmentLossMSE = BurstRecLoss()
    denoiseLossMSE = BurstRecLoss(alpha=cfg.kpn_burst_alpha,gradient_L1=~cfg.supervised)
    # denoiseLossOT = BurstResidualLoss()
    entropyLoss = EntropyLoss()

    # -=-=-=-=-=-=-=-=-=-=-
    #
    #     Noise2Noise
    #
    # -=-=-=-=-=-=-=-=-=-=-

    noise_xform = get_noise_transform(cfg.noise_params,
                                      use_to_tensor=False)

    # -=-=-=-=-=-=-=-=-=-=-
    #
    #    Final Configs
    #
    # -=-=-=-=-=-=-=-=-=-=-

    random_crop = tvT.RandomCrop(cfg.byol_patchsize)
    use_timer = False
    one = torch.FloatTensor([1.]).to(cfg.device)
    switch = True
    if use_timer:
        data_clock = Timer()
        clock = Timer()
    train_iter = iter(train_loader)
    ds_size = len(train_loader)
    small_ds = ds_size < 500
    steps_per_epoch = ds_size if not small_ds else 500

    write_examples_iter = steps_per_epoch//3
    all_filters = []

    # -=-=-=-=-=-=-=-=-=-=-
    #
    #     Start Epoch
    #
    # -=-=-=-=-=-=-=-=-=-=-

    for batch_idx in range(steps_per_epoch):

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #      Setting up for Iteration
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- setup iteration timer --
        if use_timer:
            data_clock.tic()
            clock.tic()

        # -- grab data batch --
        if small_ds and batch_idx >= ds_size: train_iter = iter(train_loader) # reset if too big
        sample = next(train_iter)
        burst,raw_img,directions = sample['burst'],sample['clean'],sample['directions']
        burst = burst.cuda(non_blocking=True)

        # -- handle possibly cached simulated bursts --
        if 'sim_burst' in sample: sim_burst = rearrange(sample['sim_burst'],'b n k c h w -> n b k c h w')
        else: sim_burst = None
        if sim_burst is None and not (cfg.n2n or cfg.supervised):
            if sim_burst is None:
                if cfg.use_kindex_lmdb: kindex = kindex_ds[batch_idx].cuda(non_blocking=True)
                else: kindex = None
                query = burst[[N//2]]
                database = torch.cat([burst[:N//2],burst[N//2+1:]])
                sim_burst = compute_similar_bursts(cfg,query,database,
                                                   cfg.sim_K,noise_level/255.,
                                                   patchsize=cfg.sim_patchsize,
                                                   shuffle_k=cfg.sim_shuffleK,
                                                   kindex=kindex,only_middle=cfg.sim_only_middle,
                                                   search_method=cfg.sim_method,db_level="frame")
        if cfg.n2n or cfg.supervised: sim_burst = burst.unsqueeze(2).repeat(1,1,2,1,1,1)
        else: sim_burst = sim_burst.cuda(non_blocking=True)

        if use_timer: data_clock.toc()
    
        # -- getting shapes of data --
        N,B,C,H,W = burst.shape
        burst = burst.cuda(non_blocking=True)
        raw_zm_img = szm(raw_img.cuda(non_blocking=True))
        burst_og = burst.clone()
        mid_img_og = burst[N//2]

        # -- shuffle over Simulated Samples --
        k_ins,k_outs = create_k_grid(sim_burst,shuffle=True)
        # k_ins,k_outs = [k_ins[0]],[k_outs[0]]

        for k_in,k_out in zip(k_ins,k_outs):
            if k_in == k_out: continue

            # -- zero gradients; ready 2 go --
            optimizer.zero_grad()
            model.zero_grad()

            # -- compute input/output data --
            if cfg.sim_only_middle:
                midi = 0 if sim_burst.shape[0] == 1 else N//2
                left_burst,right_burst = burst[:N//2],burst[N//2+1:]
                burst = torch.cat([left_burst,sim_burst[[midi],:,k_in],right_burst],dim=0)
                mid_img =  sim_burst[midi,:,k_out]
            else:
                burst = sim_burst[:,:,k_in]
                mid_img = sim_burst[N//2,:,k_out]
            # mid_img =  sim_burst[N//2,:]
            # print(burst.shape,mid_img.shape)
            # print(F.mse_loss(burst,mid_img).item())
            if cfg.supervised: gt_img = get_nmlz_img(cfg,raw_img).cuda(non_blocking=True)
            elif cfg.n2n: gt_img = noise_xform(raw_img).cuda(non_blocking=True)
            else: gt_img = mid_img
            # gt_img = torch.normal(raw_zm_img,noise_level/255.)
            
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #
            #        Dataset Augmentation
            #
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            
            # burst,gt_img = apply_transformations(burst,gt_img)

            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #
            #   Experimentally Set Hyperparams
            #
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

            # -- [before training] setting the ps and nh --
            # test_ps_nh_sizes(cfg,model,burst) 

            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #
            #      Formatting Images & FP
            #
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

            psnrs_sim = test_sim_search(cfg,burst[:,:2]+0.5,model)
            
            patches = sample_burst_patches(cfg, model, burst+0.5)
            input_patches = model.patch_helper.form_input_patches(patches)
            final_loss = model(input_patches)
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #
            #              Record Keeping
            #
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            # -- total loss --
            running_loss += final_loss.item()
            total_loss += final_loss.item()
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #
            #        Gradients & Backpropogration
            #
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            # -- compute the gradients! --
            final_loss.backward()
    
            # -- backprop now. --
            optimizer.step()
            model.update_moving_average()
            # scheduler.step()

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #            Printing to Stdout
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


        if (batch_idx % cfg.log_interval) == 0 and batch_idx > 0:

            # -- update losses --
            running_loss /= cfg.log_interval

            # -- write to stdout --
            write_info = (epoch, cfg.epochs, batch_idx, steps_per_epoch, running_loss)
            print("[%d/%d][%d/%d]: %2.3e" % write_info)

            nbatches = 2
            burst = burst[:,:nbatches] # limit batch size to run test
            psnrs_sim = test_sim_search(cfg,burst+0.5,model)
            psnrs_ftr = psnrs_sim[cfg.byol_backbone_name]
            psnrs_pix = psnrs_sim["pix"]
            print_psnr_results(psnrs_ftr,"[PSNR-ftr]")
            print_psnr_results(psnrs_pix,"[PSNR-pix]")
            print_edge_info(burst)

            # psnrs = test_sim_search(cfg,burst,model)
            # print_psnr_results(psnrs,"[PSNR-ftr]")
            # psnrs = test_sim_search_pix(cfg,burst,model)
            # print_psnr_results(psnrs,"[PSNR-pix]")

            # -- reset loss --
            running_loss = 0

        if use_timer: clock.toc()

        if use_timer:
            print("data_clock",data_clock.average_time)
            print("clock",clock.average_time)
        cfg.global_step += 1

    total_loss /= len(train_loader)
    return total_loss,record_losses

def print_edge_info(burst):

    # -- get sobel filter to detect edges --
    burstNB = rearrange(burst,'n b c h w -> (n b) c h w')
    sobel = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_t = sobel.t()
    sobel = sobel.reshape(1,3,3).repeat(3,1,1)
    sobel_t = sobel_t.reshape(1,3,3).repeat(3,1,1)
    weights = torch.stack([sobel,sobel_t],dim=0)
    weights = weights.to(burst.device)

    # -- conv --
    edges = []
    for c in range(weights.shape[0]):
        edges_c = torch.mean(F.conv2d(burstNB,weights[[c]],padding=2)).item()
        edges.append(edges_c)
    edges = np.mean(edges)
    title = "[Edges]:"
    edge_str = "%2.2e" % edges
    print(f"{title: >15} {edge_str}")

def add_color_channel(bw_pic):
    repeat = [1 for i in bw_pic.shape]
    repeat[-3] = 3
    bw_pic = bw_pic.repeat(*(repeat))
    return bw_pic

def quantize_img(cfg,image):
    params = cfg.noise_params['qis']
    pix_max = 2**params['nbits'] - 1
    image += 0.5
    image *= params['alpha']
    image = torch.round(image)
    image = torch.clamp(image, 0, pix_max)
    image /= params['alpha']
    image -= 0.5
    return image
    
def get_nmlz_img(cfg,raw_img):
    pix_max = 2**3-1
    noise_type = cfg.noise_params.ntype
    if noise_type in ["g","hg"]: nmlz_raw = raw_img - raw_offset
    elif noise_type in ["qis"]:
        params = cfg.noise_params[noise_type]
        pix_max = 2**params['nbits'] - 1
        raw_img_bw = tvF.rgb_to_grayscale(raw_img,1)
        raw_img_bw = add_color_channel(raw_img_bw)
        # nmlz_raw = raw_scale * raw_img_bw - 0.5
        raw_img_bw *= params['alpha']
        raw_img_bw = torch.round(raw_img_bw)
        # print("ll",ll_pic.min().item(),ll_pic.max().item())
        raw_img_bw = torch.clamp(raw_img_bw, 0, pix_max)
        raw_img_bw /= params['alpha']
        # -- end of qis noise transform --

        # -- start dnn normalization for optimization --
        nmlz_raw = raw_img_bw - 0.5
    else:
        print("[Warning]: Check normalize raw image.")        
        nmlz_raw = raw_img
    return nmlz_raw

def test_loop(cfg,model,test_data,test_loader,epoch,writer=None):
    model.eval()
    model = model.to(cfg.device)
    noise_type = cfg.noise_params.ntype
    total_psnr = 0
    total_loss = 0
    use_record = False
    record_test = pd.DataFrame({'psnr':[]})
    test_iter = iter(test_loader)
    num_batches,D = cfg.byol_num_test_samples,len(test_iter) 
    num_batches = num_batches if D > num_batches else D

    # -- setup results --
    results = edict()
    results.ftr,results.pix = edict(),edict()
    noise_names = get_keys_noise_level_grid(cfg)
    for name in noise_names:
        results.ftr[name] = edict()
        results.ftr[name].psnrs = []
        results.pix[name] = edict()
        results.pix[name].psnrs = []

    print("Testing...")
    with torch.no_grad():
        for sample_idx in range(num_batches):

            sample = test_data[sample_idx]
            burst,raw_img,directions = sample['burst'],sample['clean'],sample['directions']
            burst,raw_img = burst[:,None],raw_img[None,:]
            B = raw_img.shape[0]

            # -- reshaping of data --
            raw_img = raw_img.cuda(non_blocking=True)
            burst = burst.cuda(non_blocking=True)

            # -- run testing --
            psnrs_sim = test_sim_search_serial_batch(cfg,burst,model)
            psnrs_ftr = psnrs_sim[cfg.byol_backbone_name]
            psnrs_pix = psnrs_sim["pix"]
            # print_psnr_results(psnrs_ftr,"[PSNR-ftr]")
            # print_psnr_results(psnrs_pix,"[PSNR-pix]")
                
            for name in results.pix.keys():
                results.ftr[name].psnrs.extend(psnrs_ftr[name].psnrs)
                results.pix[name].psnrs.extend(psnrs_pix[name].psnrs)

    pix_ftr = ['pix','ftr']
    for name in results.pix.keys():
        for elem in pix_ftr:
            results[elem][name].ave = np.mean(results[elem][name].psnrs)
            results[elem][name].std = np.std(results[elem][name].psnrs)
            results[elem][name].min = np.min(results[elem][name].psnrs)
            results[elem][name].max = np.max(results[elem][name].psnrs)

    print(f"[TESTING: {cfg.current_epoch}]")
    print_psnr_results(results.ftr,"[PSNR-ftr]")
    print_psnr_results(results.pix,"[PSNR-pix]")
    psnr_ave = results.ftr.clean.ave
    return psnr_ave,results


def write_input_output(cfg,model,burst,aligned,denoised,filters,directions):

    """
    :params burst: input images to the model, :shape [B, N, C, H, W]
    :params aligned: output images from the alignment layers, :shape [B, N, C, H, W]
    :params denoised: output images from the denoiser, :shape [B, N, C, H, W]
    :params filters: filters used by model, :shape [B, L, N, K2, 1, Hf, Wf] with Hf = (H or 1) for L = number of cascaded filters
    """

    # -- file path --
    path = Path(f"./output/n2sim/io_examples/{cfg.exp_name}/")
    if not path.exists(): path.mkdir(parents=True)

    # -- init --
    B,N,C,H,W = burst.shape

    # -- save histogram of residuals --
    denoised_np = denoised.detach().cpu().numpy()
    plot_histogram_residuals_batch(denoised_np,cfg.global_step,path,rand_name=False)

    # -- save histogram of gradients (denoiser) --
    if not model.use_unet_only:
        denoiser = model.denoiser_info.model
        plot_histogram_gradients(denoiser,"denoiser",cfg.global_step,path,rand_name=False)

    # -- save histogram of gradients (alignment) --
    if model.use_alignment:
        alignment = model.align_info.model
        plot_histogram_gradients(alignment,"alignment",cfg.global_step,path,rand_name=False)

    # -- save gradient norm by layer (denoiser) --
    if not model.use_unet_only:
        denoiser = model.denoiser_info.model
        plot_histogram_gradient_norms(denoiser,"denoiser",cfg.global_step,path,rand_name=False)

    # -- save gradient norm by layer (alignment) --
    if model.use_alignment:
        alignment = model.align_info.model
        plot_histogram_gradient_norms(alignment,"alignment",cfg.global_step,path,rand_name=False)

    if B > 4: B = 4
    for b in range(B):
        
        # -- save dirty & clean & res triplet --
        fn = path / Path(f"image_{cfg.global_step}_{b}.png")
        res = burst[b][N//2] - denoised[b].mean(0)
        imgs = torch.stack([burst[b][N//2],denoised[b].mean(0),res],dim=0)
        tv_utils.save_image(imgs,fn,nrow=3,normalize=True,range=(-0.5,0.5))
        
        # -- save images --
        fn = path / Path(f"{cfg.global_step}_{b}.png")
        burst_b = torch.cat([burst[b][[N//2]] - burst[b][[0]],burst[b],burst[b][[N//2]] - burst[b][[-1]]],dim=0)
        aligned_b = torch.cat([aligned[b][[N//2]] - aligned[b][[0]],aligned[b],aligned[b][[N//2]] - aligned[b][[-1]]],dim=0)
        denoised_b = torch.cat([denoised[b][[N//2]] - denoised[b][[0]],denoised[b],denoised[b][[N//2]] - denoised[b][[-1]]],dim=0)
        imgs = torch.cat([burst_b,aligned_b,denoised_b],dim=0) # 2N,C,H,W
        tv_utils.save_image(imgs,fn,nrow=N+2,normalize=True,range=(-0.5,0.5))

        # -- save filters --
        fn = path / Path(f"filters_{cfg.global_step}_{b}.png")
        K = int(np.sqrt(filters.shape[3]))
        L = filters.shape[1]
        if filters.shape[-1] > 1:
            S = npr.permutation(filters.shape[-1])[:10]
            filters_b = filters[b,...,0,S,S].view(N*10*L,1,K,K)
        else: filters_b = filters[b,...,0,0,0].view(N*L,1,K,K)
        tv_utils.save_image(filters_b,fn,nrow=N,normalize=True)

        # -- save direction image --
        fn = path / Path(f"arrows_{cfg.global_step}_{b}.png")
        if len(directions[b]) > 1 and len(directions[b].shape) > 1:
            arrows = create_arrow_image(directions[b],pad=2)
            tv_utils.save_image([arrows],fn)


    print(f"Wrote example images to file at [{path}]")
    plt.close("all")



def create_arrow_image(directions,pad=2):
    D = len(directions)
    assert D == 1,f"Only one direction right now. Currently, [{len(directions)}]"
    W = 100
    S = (W + pad) * D + pad
    arrows = np.zeros((S,W+2*pad,3))
    direction = directions[0]
    for i in range(D):
        col_i = (pad+W)*i+pad
        canvas = arrows[col_i:col_i+W,pad:pad+W,:]
        start_point = (0,0)
        x_end = direction[0].item()
        y_end = direction[1].item()
        end_point = (x_end,y_end)

        fig = Figure(dpi=300)
        plt_canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.annotate("",
                    xy=end_point, xycoords='data',
                    xytext=start_point, textcoords='data',
                    arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),
        )
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        plt_canvas.draw()       # draw the canvas, cache the renderer
        canvas = np.array(plt_canvas.buffer_rgba())[:,:,:]
        arrows = canvas
    arrows = torch.Tensor(arrows.astype(np.uint8)).transpose(0,2).transpose(1,2)
    return arrows


