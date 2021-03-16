
# -- python imports --
import bm3d,uuid,cv2
import numpy as np
import numpy.random as npr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from einops import rearrange, repeat, reduce

# -- just to draw an fing arrow --
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


# -- pytorch imports --
import torch
import torch.nn.functional as F
import torchvision.utils as tv_utils

# -- project code --
import settings
from pyutils.timer import Timer
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr
from datasets.transforms import ScaleZeroMean
from layers.ot_pytorch import sink_stabilized
from n2nwl.plot import plot_histogram_residuals_batch,plot_histogram_gradients,plot_histogram_gradient_norms

# -- [local] project code --
from .utils import init_record

def train_loop(cfg,model,optimizer,criterion,train_loader,epoch):
    return train_loop_offset(cfg,model,optimizer,criterion,train_loader,epoch)

def test_loop(cfg,model,criterion,test_loader,epoch):
    return test_loop_offset(cfg,model,criterion,test_loader,epoch)


def train_loop_offset(cfg,model,optimizer,criterion,train_loader,epoch):
    model.train()
    model = model.to(cfg.device)
    N = cfg.N
    total_loss = 0
    running_loss = 0
    sf_losses,sf_count = 0,0
    kl_losses,kl_count = 0,0
    temporal_losses,temporal_count = 0,0
    write_examples = True
    write_examples_iter = 800
    szm = ScaleZeroMean()
    record = init_record()
    use_record = False

    # if cfg.N != 5: return
    for batch_idx, (burst_imgs, res_imgs, raw_img, directions) in enumerate(train_loader):

        optimizer.zero_grad()
        model.zero_grad()

        # fig,ax = plt.subplots(figsize=(10,10))
        # imgs = burst_imgs + 0.5
        # imgs.clamp_(0.,1.)
        # raw_img = raw_img.expand(burst_imgs.shape)
        # print(imgs.shape,raw_img.shape)
        # all_img = torch.cat([imgs,raw_img],dim=1)
        # print(all_img.shape)
        # grids = [tv_utils.make_grid(all_img[i],nrow=16) for i in range(cfg.dynamic.frames)]
        # ims = [[ax.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in grids]
        # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
        # ani.save(f"{settings.ROOT_PATH}/train_loop_voc.mp4", writer=writer)
        # print("I DID IT!")
        # return

        # -- reshaping of data --
        # raw_img = raw_img.cuda(non_blocking=True)
        input_order = np.arange(cfg.N)
        # print("pre",input_order,cfg.blind,cfg.N)
        middle_img_idx = -1
        if not cfg.input_with_middle_frame:
            middle = len(input_order) // 2
            # print(middle)
            middle_img_idx = input_order[middle]
            # input_order = np.r_[input_order[:middle],input_order[middle+1:]]
        else:
            middle = len(input_order) // 2
            input_order = np.arange(cfg.N)
            middle_img_idx = input_order[middle]
            # input_order = np.arange(cfg.N)
        # print("post",input_order,cfg.blind,cfg.N,middle_img_idx)

        burst_imgs = burst_imgs.cuda(non_blocking=True)
        # print(cfg.N,cfg.blind,[input_order[x] for x in range(cfg.input_N)])
        # stacked_burst = torch.cat([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
        # print("stacked_burst",stacked_burst.shape)
        # print("burst_imgs.shape",burst_imgs.shape)
        # print("stacked_burst.shape",stacked_burst.shape)

        # -- add input noise --
        burst_imgs_noisy = burst_imgs.clone()
        if cfg.input_noise:
            noise = np.random.rand() * cfg.input_noise_level
            if cfg.input_noise_middle_only:
                burst_imgs_noisy[middle_img_idx] = torch.normal(burst_imgs_noisy[middle_img_idx],noise)
            else:
                burst_imgs_noisy = torch.normal(burst_imgs_noisy,noise)

        # -- create inputs for kpn --
        stacked_burst = torch.stack([burst_imgs_noisy[input_order[x]] for x in range(cfg.input_N)],dim=1)
        cat_burst = torch.cat([burst_imgs_noisy[input_order[x]] for x in range(cfg.input_N)],dim=1)
        # print(stacked_burst.shape)
        # print(cat_burst.shape)

        # -- extract target image --
        mid_img =  burst_imgs[middle_img_idx]
        raw_img_zm = szm(raw_img.cuda(non_blocking=True))
        if cfg.supervised: t_img = szm(raw_img.cuda(non_blocking=True))
        else: t_img = burst_imgs[middle_img_idx]
        
        # -- direct denoising --
        mis_ave = torch.mean(stacked_burst,dim=1)
        # aligned,rec_img,temporal_loss,filters = model(cat_burst,stacked_burst)
        aligned,rec_img,filters = model(cat_burst,stacked_burst)
        temporal_loss = torch.FloatTensor([-1.]).to(cfg.device)

        # print("(a) [m: %2.2e] [std: %2.2e] vs [tgt: %2.2e]" % (torch.mean(mid_img - raw_img_zm).item(),F.mse_loss(mid_img,raw_img_zm).item(),(25./255)**2) )
        # r_raw_img_zm = raw_img_zm.unsqueeze(1).repeat(1,N,1,1,1)
        # print("(b) [m: %2.2e] [std: %2.2e] vs [tgt: %2.2e]" % (torch.mean(aligned - r_raw_img_zm).item(),F.mse_loss(aligned,r_raw_img_zm).item(),(25./255)**2) )

        # -- compare with stacked burst --
        # print(cfg.blind,t_img.min(),t_img.max(),t_img.mean())
        # rec_img = rec_img.expand(t_img.shape)
        # loss = F.mse_loss(t_img,rec_img)

        # -- sparse filter loss (sf_loss) --
        # sf_loss = sparse_filter_loss(filters)
        sf_loss = torch.FloatTensor([-1.]).to(cfg.device)

        # -- compute loss to optimize --
        losses = criterion(aligned, rec_img, t_img, cfg.global_step)
        loss = np.sum(losses) #+ sf_loss + temporal_loss
        # loss = losses[1]
        kpn_loss = loss
        kpn_coeff = 1. # .9997**cfg.global_step
        # temporal_loss = temporal_loss.item()
        # mse_loss = F.mse_loss(rec_img,mid_img)

        # -- compute ot loss to optimize --
        # residuals = aligned - rec_img.unsqueeze(1).repeat(1,N,1,1,1)
        # residuals = rearrange(residuals,'b n c h w -> b n (h w) c')
        # ot_loss = ot_pairwise_bp(residuals,reg=1.0,K=5)
        # ot_coeff = 1 - .997**cfg.global_step

        # -- compute kl loss to optimize -- 
        if cfg.supervised: kl_ref = szm(raw_img.cuda(non_blocking=True))
        else: kl_ref = rec_img
        residuals = aligned - kl_ref.unsqueeze(1).repeat(1,N,1,1,1)
        residuals = rearrange(residuals,'b n c h w -> b n (h w) c')
        kl_loss = kl_pairwise_bp(residuals,K=100,supervised=cfg.supervised)
        kl_coeff = 100# - .997**cfg.global_step
        # kl_loss = torch.FloatTensor([-1.]).to(cfg.device)

        # -- final loss --
        # loss = ot_coeff * ot_loss + kpn_loss
        # loss = kl_coeff * kl_loss + kpn_coeff * kpn_loss
        loss = kpn_coeff * kpn_loss
            
        # -- update info --
        running_loss += loss.item()
        total_loss += loss.item()

        # -- update sparse filter loss info --
        sf_losses += sf_loss.item()
        sf_count += 1

        # -- update temporal loss info --
        temporal_losses += temporal_loss.item()
        temporal_count += 1

        # -- update temporal loss info --
        kl_losses += kl_loss.item()
        kl_count += 1


        # -- BP and optimize --
        loss.backward()
        optimizer.step()

        if (batch_idx % cfg.log_interval) == 0 and batch_idx > 0:

            # -- compute mse for [rec img] --
            BS = raw_img.shape[0]            
            raw_img = raw_img.cuda(non_blocking=True)
            mse_loss = F.mse_loss(raw_img,rec_img+0.5,reduction='none').reshape(BS,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr_ave = np.mean(mse_to_psnr(mse_loss))
            psnr_std = np.std(mse_to_psnr(mse_loss))
            running_loss /= cfg.log_interval

            # -- psnr for misaligned ave --
            mse_loss = F.mse_loss(raw_img,mis_ave+0.5,reduction='none').reshape(BS,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            mis_psnr_ave = np.mean(mse_to_psnr(mse_loss))
            mis_psnr_std = np.std(mse_to_psnr(mse_loss))

            # -- psnr for [bm3d] --
            bm3d_nb_psnrs = []
            for b in range(BS):
                bm3d_rec = bm3d.bm3d(mid_img[b].cpu().transpose(0,2)+0.5, sigma_psd=25/255, stage_arg=bm3d.BM3DStages.ALL_STAGES)
                bm3d_rec = torch.FloatTensor(bm3d_rec).transpose(0,2)
                b_loss = F.mse_loss(raw_img[b].cpu(),bm3d_rec,reduction='none').reshape(BS,-1)
                b_loss = torch.mean(b_loss,1).detach().cpu().numpy()
                bm3d_nb_psnr = np.mean(mse_to_psnr(b_loss))
                bm3d_nb_psnrs.append(bm3d_nb_psnr)
            bm3d_nb_ave = np.mean(bm3d_nb_psnrs)
            bm3d_nb_std = np.std(bm3d_nb_psnrs)

            # -- temporal loss --
            ave_temporal_loss = temporal_losses / temporal_count if temporal_count > 0 else 0
            temporal_losses,temporal_count = 0,0

            # -- sparse filter loss --
            ave_sf_loss = sf_losses / sf_count if sf_count > 0 else 0
            sf_losses,sf_count = 0,0

            # -- kl loss --
            ave_kl_loss = kl_losses / kl_count if kl_count > 0 else 0
            kl_losses,kl_count = 0,0

            # -- write to stdout --
            write_info = (epoch, cfg.epochs, batch_idx,len(train_loader),running_loss,psnr_ave,psnr_std,bm3d_nb_ave,bm3d_nb_std,
                          mis_psnr_ave,mis_psnr_std,ave_temporal_loss,ave_sf_loss,ave_kl_loss)
            print("[%d/%d][%d/%d]: %2.3e [PSNR]: %2.2f +/- %2.2f [bm3d]: %2.2f +/- %2.2f [misaligned]: %2.2f +/- %2.2f [loss-t]: %.2e [loss-sf]: %.2e [loss-kl]: %.2e" % write_info)
            # print("[%d/%d][%d/%d]: %2.3e [PSNR]: %2.2f +/- %2.2f"%(epoch, cfg.epochs, batch_idx,
            #                                                        len(train_loader),
            #                                                        running_loss,psnr_ave,psnr_std))
            running_loss = 0

            # -- record information --
            if use_record:
                rec = rec_img
                raw = raw_img_zm
                frame_results = compute_ot_frame(aligned,rec,raw,reg=0.5)
                burst_results = compute_ot_burst(aligned,rec,raw,reg=0.5)
                psnr_record = {'psnr_ave':psnr_ave,'psnr_std':psnr_std}
                kpn_record = {'kpn_loss':kpn_loss}
                new_record = merge_records(frame_results,burst_results,psnr_record,kpn_record)
                record = record.append(new_record,ignore_index=True)

        # -- write examples --
        if write_examples and (batch_idx % write_examples_iter) == 0:
            write_input_output(cfg,model,stacked_burst,aligned,filters,directions)

        cfg.global_step += 1
    total_loss /= len(train_loader)
    return total_loss,record

def test_loop_offset(cfg,model,criterion,test_loader,epoch):
    model.eval()
    model = model.to(cfg.device)
    total_psnr = 0
    total_loss = 0
    psnrs = np.zeros( (len(test_loader),cfg.batch_size) )
    szm = ScaleZeroMean()

    with torch.no_grad():
        for batch_idx, (burst_imgs, res_imgs, raw_img, directions) in enumerate(test_loader):
            BS = raw_img.shape[0]
            
            # -- selecting input frames --
            input_order = np.arange(cfg.N)
            # print("pre",input_order)
            middle_img_idx = -1
            if not cfg.input_with_middle_frame:
                middle = cfg.N // 2
                # print(middle)
                middle_img_idx = input_order[middle]
                # input_order = np.r_[input_order[:middle],input_order[middle+1:]]
            else:
                middle = len(input_order) // 2
                input_order = np.arange(cfg.N)
                middle_img_idx = input_order[middle]
                # input_order = np.arange(cfg.N)
            
            # -- reshaping of data --
            raw_img = raw_img.cuda(non_blocking=True)
            burst_imgs = burst_imgs.cuda(non_blocking=True)
            stacked_burst = torch.stack([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
            cat_burst = torch.cat([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
    
            # -- extract images for psnr --
            mid_img =  burst_imgs[middle_img_idx]
            raw_img_zm = szm(raw_img.cuda(non_blocking=True))

            # -- denoising --
            rec_img = model(cat_burst,stacked_burst)[1].detach()

            # if not cfg.input_with_middle_frame:
            #     rec_img = model(cat_burst,stacked_burst)[1]
            # else:
            #     rec_img = model(cat_burst,stacked_burst)[0][middle_img_idx]

            # rec_img = burst_imgs[middle_img_idx] - rec_res
            
            # -- compare with stacked targets --
            rec_img = rescale_noisy_image(rec_img)        

            # -- compute psnr --
            loss = F.mse_loss(raw_img,rec_img,reduction='none').reshape(BS,-1)
            # loss = F.mse_loss(raw_img,burst_imgs[cfg.input_N//2]+0.5,reduction='none').reshape(BS,-1)
            loss = torch.mean(loss,1).detach().cpu().numpy()
            psnr = mse_to_psnr(loss)
            psnrs[batch_idx,:] = psnr
                        
            total_psnr += np.mean(psnr)
            total_loss += np.mean(loss)

            # if (batch_idx % cfg.test_log_interval) == 0:
            #     root = Path(f"{settings.ROOT_PATH}/output/n2n/offset_out_noise/rec_imgs/N{cfg.N}/e{epoch}")
            #     if not root.exists(): root.mkdir(parents=True)
            #     fn = root / Path(f"b{batch_idx}.png")
            #     nrow = int(np.sqrt(cfg.batch_size))
            #     rec_img = rec_img.detach().cpu()
            #     grid_imgs = tv_utils.make_grid(rec_img, padding=2, normalize=True, nrow=nrow)
            #     plt.imshow(grid_imgs.permute(1,2,0))
            #     plt.savefig(fn)
            #     plt.close('all')

            if (batch_idx % cfg.test_log_interval) == 0:
                print("[%d/%d] Running Test PSNR: %2.2f" % (batch_idx, len(test_loader), total_psnr / (batch_idx+1)))

    psnr_ave = np.mean(psnrs)
    psnr_std = np.std(psnrs)
    ave_loss = total_loss / len(test_loader)
    print("[N: %d] Testing: [psnr: %2.2f +/- %2.2f] [ave loss %2.3e]"%(cfg.N,psnr_ave,psnr_std,ave_loss))
    return psnr_ave



def ot_pairwise_bp(residuals,reg=0.5,K=3):
    """
    :param residuals: shape [B N D C]
    """
    
    # -- init --
    B,N,D,C = residuals.shape

    # -- create triplets
    S = B*K
    indices,S = create_loop_indices(B,N,S)

    # -- compute losses --
    ot_loss = 0
    for (bi,bj,i,j) in indices:
        ri,rj = residuals[bi,i],residuals[bj,j]
        M = torch.sum(torch.pow(ri.unsqueeze(1) - rj,2),dim=-1)
        loss = sink_stabilized(M,reg)
        weight = ( torch.mean(ri) + torch.mean(rj) ) / 2.
        ot_loss += loss * weight.item()
    return ot_loss / len(indices)


def compute_ot_frame(aligned,rec,raw,reg=0.5):
    # -- init --
    B,N,C,H,W = aligned.shape
    aligned = aligned.detach()
    rec = rec.detach()

    # -- rec residuals --
    rec_res = aligned - rec.unsqueeze(1).repeat(1,N,1,1,1)
    rec_res = rearrange(rec_res,'b n c h w -> b n (h w) c')
    ot_loss_rec,ot_loss_rec_mid,ot_loss_rec_w = ot_frame_pairwise(rec_res,reg=reg)

    # -- raw residuals --
    raw_res = aligned - raw.unsqueeze(1).repeat(1,N,1,1,1)
    raw_res = rearrange(raw_res,'b n c h w -> b n (h w) c')
    ot_loss_raw,ot_loss_raw_mid,ot_loss_raw_w = ot_frame_pairwise(raw_res,reg=reg)

    # -- format results --
    results = {'ot_loss_rec_frame':ot_loss_rec,
               'ot_loss_rec_frame_mid':ot_loss_rec_mid,
               'ot_loss_rec_frame_w':ot_loss_rec_w,
               'ot_loss_raw_frame':ot_loss_raw,
               'ot_loss_raw_frame_mid':ot_loss_raw_mid,
               'ot_loss_raw_frame_w':ot_loss_raw_w,}

    return results

def ot_frame_pairwise(residuals,reg=0.5):
    """
    :paraam residuals: shape [B N D C]
    """
    
    # -- init --
    B,N,D,C = residuals.shape

    # -- create pairs --
    pairs,S = select_pairs(N)

    # -- compute losses --
    ot_loss,ot_loss_mid,ot_loss_w = 0,0,0
    for b in range(B):
        for (i,j) in pairs:
            ri,rj = residuals[b,i],residuals[b,j]
            M = torch.sum(torch.pow(ri.unsqueeze(1) - rj,2),dim=-1)
            loss = sink_stabilized(M,reg)
            ot_loss += loss
            weight = (torch.sum(ri**2) + torch.sum(rj**2)).item()
            ot_loss_w += weight * loss
            if i == N//2 or j == N//2:
                ot_loss_mid += loss
    return ot_loss,ot_loss_mid,ot_loss_w
    

def compute_ot_burst(aligned,rec,raw,reg=0.5):
    # -- init --
    B,N,C,H,W = aligned.shape
    aligned = aligned.detach()
    rec = rec.detach()

    # -- rec residuals --
    rec_res = aligned - rec.unsqueeze(1).repeat(1,N,1,1,1)
    rec_res = rearrange(rec_res,'b n c h w -> b (n h w) c')
    ot_loss_rec,ot_loss_rec_w = ot_burst_pairwise(rec_res,reg=reg)

    # -- raw residuals --
    raw_res = aligned - raw.unsqueeze(1).repeat(1,N,1,1,1)
    raw_res = rearrange(raw_res,'b n c h w -> b (n h w) c')
    ot_loss_raw,ot_loss_raw_w = ot_burst_pairwise(raw_res,reg=reg)

    # -- format results --
    results = {'ot_loss_rec_burst':ot_loss_rec,'ot_loss_raw_burst':ot_loss_raw,
               'ot_loss_rec_burst_w':ot_loss_rec_w,'ot_loss_raw_burst_w':ot_loss_raw_w,}

    return results

def ot_burst_pairwise(residuals,reg=0.5):
    """
    :param residuals: shape [B D C]
    """
    # -- init --
    B,D,C = residuals.shape

    # -- create pairs --
    pairs,S = select_pairs(B)

    # -- compute losses --
    ot_loss,ot_loss_w = 0,0
    for (i,j) in pairs:
        ri,rj = residuals[i],residuals[j]
        M = torch.sum(torch.pow(ri.unsqueeze(1) - rj,2),dim=-1)
        loss = sink_stabilized(M,reg)
        ot_loss += loss
        weight = (torch.sum(ri**2) + torch.sum(rj**2)).item()
        ot_loss_w += weight * loss
    return ot_loss,ot_loss_w

def select_pairs( N, S = None, skip_middle = False):
    if skip_middle:
        pairs = list(set([(i,j) for i in range(N) for j in range(N) if i != N//2 and j != N//2]))
    else:
        pairs = list(set([(i,j) for i in range(N) for j in range(N)]))
    P = len(pairs)
    if S is None: S = P
    r_idx = npr.choice(range(P),S)
    s_pairs = [pairs[idx] for idx in r_idx]
    return s_pairs,S

def create_middle_loop_indices(B,N,S):
    indices = []
    for i in range(N):
        for bi in range(B):
            for bj in range(B):
                if bi > bj: continue
                index = (bi,bj,i,N//2)
                indices.append(index)

    P = len(indices)
    indices = list(set(indices))
    assert P == len(indices), "We only created the list with unique elements"
    if S is None: S = P
    if S > P: r_idx = npr.choice(range(P),P)
    else: r_idx = npr.choice(range(P),S)
    s_indices = [indices[idx] for idx in r_idx]
    return s_indices,S

def create_no_middle_loop_indices(B,N,S):
    indices = []
    for i in range(N):
        for j in range(N):
            if i > j: continue
            if i == N//2 or j == N//2: continue
            for bi in range(B):
                for bj in range(B):
                    if bi > bj: continue
                    index = (bi,bj,i,j)
                    indices.append(index)

    P = len(indices)
    indices = list(set(indices))
    assert P == len(indices), "We only created the list with unique elements"
    if S is None: S = P
    if S > P: r_idx = npr.choice(range(P),P)
    else: r_idx = npr.choice(range(P),S)
    s_indices = [indices[idx] for idx in r_idx]
    return s_indices,S
    
def create_loop_indices(B,N,S):
    indices = []
    for i in range(N):
        for j in range(N):
            if i > j: continue
            for bi in range(B):
                for bj in range(B):
                    if bi > bj: continue
                    index = (bi,bj,i,j)
                    indices.append(index)

    P = len(indices)
    indices = list(set(indices))
    assert P == len(indices), "We only created the list with unique elements"
    if S is None: S = P
    if S > P: r_idx = npr.choice(range(P),P)
    else: r_idx = npr.choice(range(P),S)
    s_indices = [indices[idx] for idx in r_idx]
    return s_indices,S

def create_triplets(pairs,B):
    triplets = []
    for b in range(B):
        for (i,_) in pairs:
            for (_,j) in pairs:
                triplets.append([b,i,j])
    return triplets

def merge_records(*args):
    record = {}
    for d in args:
        record.update(d)
    return record

def write_input_output(cfg,model,burst,aligned,filters,directions):

    """
    :params burst: input images to the model, :shape [B, N, C, H, W]
    :params aligned: output images from the model, :shape [B, N, C, H, W]
    :params filters: filters used by model, :shape [B, N, K2, 1, Hf, Wf] with Hf = (H or 1)
    """

    # -- file path --
    path = Path(f"./output/n2n-kpn/io_examples/{cfg.exp_name}/")
    if not path.exists(): path.mkdir(parents=True)

    # -- init --
    B,N,C,H,W = burst.shape

    # -- save histogram of residuals --
    denoised_np = aligned.detach().cpu().numpy()
    plot_histogram_residuals_batch(denoised_np,cfg.global_step,path,rand_name=False)

    # -- save histogram of gradients --
    plot_histogram_gradients(model,cfg.global_step,path,rand_name=False)

    # -- save gradient norm by layer --
    plot_histogram_gradient_norms(model,cfg.global_step,path,rand_name=False)

    # -- save file per burst --
    for b in range(B):
        
        # -- save images --
        fn = path / Path(f"{cfg.global_step}_{b}.png")
        burst_b = torch.cat([burst[b][[N//2]] - burst[b][[0]],burst[b],burst[b][[N//2]] - burst[b][[-1]]],dim=0)
        aligned_b = torch.cat([aligned[b][[N//2]] - aligned[b][[0]],aligned[b],aligned[b][[N//2]] - aligned[b][[-1]]],dim=0)
        imgs = torch.cat([burst_b,aligned_b],dim=0) # 2N,C,H,W
        tv_utils.save_image(imgs,fn,nrow=N+2,normalize=True,range=(-0.5,0.5))

        # -- save filters --
        fn = path / Path(f"filters_{cfg.global_step}_{b}.png")
        K = int(np.sqrt(filters.shape[2]))
        if filters.shape[-1] > 1:
            S = npr.permutation(filters.shape[-1])[:10]
            filters_b = filters[b,:,:,0,S,S].view(N*10,1,K,K)
        else: filters_b = filters[b,:,:,0,0,0].view(N,1,K,K)
        tv_utils.save_image(filters_b,fn,nrow=N,normalize=True)

        # -- save direction image --
        fn = path / Path(f"arrows_{cfg.global_step}_{b}.png")
        arrows = create_arrow_image(directions[b],pad=2)
        tv_utils.save_image([arrows],fn)

    plt.close("all")
    print(f"Wrote example images to file at [{path}]")


def create_arrow_image(directions,pad=2):
    D = len(directions)
    assert D == 1,"Only one direction right now."
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

def sparse_filter_loss(filters):
    """
    :params filters: shape: [B,N,K2,1,H,W] with H = 1 or H
    """

    # -- init --
    B,N,K2 = filters.shape[:3]
    filters = filters.view(B,N,K2)
    afilters = torch.abs(filters)

    # -- standard l1 loss --
    l1_filters = torch.sum(afilters)

    # -- penalize filter values not close to 1 --
    to1_filters = torch.mean(torch.abs(torch.sum(afilters,dim=2) - 1))

    # -- old sparse filter loss --
    # afilters = torch.abs(filters)
    # sfilters = torch.mean(afilters,dim=2)
    # zero = torch.FloatTensor([0.]).to(filters.device)
    # wfilters = torch.where(sfilters > float(1.), sfilters, zero)
    
    loss = l1_filters + to1_filters
    return loss

def kl_pairwise_bp(residuals,bins=255,K=10,supervised=True):
    """
    :param residuals: shape [B N D C]
    """
    
    # -- init --
    B,N,D,C = residuals.shape

    # -- create triplets
    S = B*K
    indices,S = create_loop_indices(B,N,S)
    # indices,S = create_no_middle_loop_indices(B,N,S)
    # if supervised:
    #     indices,S = create_middle_loop_indices(B,N,S)
    # else:
    #     indices,S = create_no_middle_loop_indices(B,N,S)
        # indices,S = create_loop_indices(B,N,S)
    
    # -- compute losses --
    kl_loss = 0
    for (bi,bj,i,j) in indices:
        ri,rj = residuals[bi,i],residuals[bj,j]
        kl_loss += compute_binned_kl(ri,rj,bins=255)
    return kl_loss / len(indices)

def compute_binned_kl(ri,rj,bins=255):
    """
    compute empirical kl for binned residuals
    """
    ri,rj = ri.reshape(-1),rj.reshape(-1)

    amin = min([ri.min(),rj.min()])
    ri -= amin
    rj -= amin
    amax = max([ri.max(),rj.max()])
    scale = bins/amax

    ri = (scale*ri).int()
    rj = (scale*rj).int()

    cri = torch.bincount(ri).float()
    crj = torch.bincount(rj).float()

    cri /= cri.sum()
    crj /= crj.sum()
    
    si = int(cri.size()[0])
    sj = int(crj.size()[0])

    if si > sj: cri = cri[:sj]
    elif si < sj: crj = crj[:si]
    args = torch.where(torch.logical_and(cri,crj))[0]
    kl = 0
    for index in args:
        freq_index_i = cri[index]
        freq_index_j = crj[index]
        kl += freq_index_i * (torch.log(freq_index_i) - torch.log(freq_index_j))
    return kl

