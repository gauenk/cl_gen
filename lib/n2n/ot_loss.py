
# -- python imports --
import bm3d,uuid,cv2
import pandas as pd
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
from pyutils import np_log,rescale_noisy_image,mse_to_psnr
from datasets.transforms import ScaleZeroMean
from layers.ot_pytorch import sink_stabilized
from pyutils.plot import add_legend

# -- [local] project code --
from n2n.utils import init_record


def run_test_aburst(cfg,criterion,train_loader):

    reg = 1.0
    for batch_idx, (burst, residuals, raw_img, directions) in enumerate(train_loader):

        # -- init --
        N,B,C,H,W = burst.shape
        burst = burst.cuda(non_blocking=True)
        raw_img = raw_img.cuda(non_blocking=True) - 0.5
        
        # -- compute residuals --
        misaligned_residual = rearrange(burst - raw_img.unsqueeze(0).repeat(N,1,1,1,1),'n b c h w -> b n (h w) c')

        # -- save img --
        # res = burst - raw_img.unsqueeze(0).repeat(N,1,1,1,1)
        # res = rearrange(torch.cat([res,raw_img.unsqueeze(0)],dim=0),'n b c h w -> b n c h w')
        # burst_r = rearrange(torch.cat([burst,raw_img.unsqueeze(0)],dim=0),'n b c h w -> b n c h w')
        # img = rearrange(torch.cat([burst_r,res],dim=1),'b n c h w -> (b n) c h w')
        # fn = "./test_ot_loss.png"
        # tv_utils.save_image(img,fn,nrow=B,normalize=True)
        # print(f"Wrote file to {fn}")
        
        print("25/255 = {:.2e}".format(25/255))
        for b in range(B):
            for n in range(N):
                print("({:d},{:d}): {:.2e} +/- {:.2e}".format(b,n,misaligned_residual[b,n].mean(),misaligned_residual[b,n].std()))

        # -- compute ot for pairwise of misaligned images on sides --
        ot_02 = 0
        for b in range(B): ot_02 += compute_pair_ot(misaligned_residual[b,0],misaligned_residual[b,2],reg)
        ot_02 /= B

        # -- compute ot for pairwise of misaligned images to middle --
        ot_01 = 0
        for b in range(B): ot_01 += compute_pair_ot(misaligned_residual[b,0],misaligned_residual[b,N//2],reg)
        ot_01 /= B

        # -- compute ot for pairwise of misaligned images to middle --
        ot_12 = 0
        for b in range(B): ot_12 += compute_pair_ot(misaligned_residual[b,2],misaligned_residual[b,N//2],reg)
        ot_12 /= B

        # -- compute ot for pairwise of misaligned images to middle --
        g1 = torch.normal(torch.zeros((64*64*3)),25/255).view(-1,3)
        g2 = torch.normal(torch.zeros((64*64*3)),25/255).view(-1,3)
        g1,g2 = g1.cuda(non_blocking=True),g2.cuda(non_blocking=True)
        ot_gg = compute_pair_ot(g1,g2,reg)

        # -- compute ot for misaligned images to gaussian --
        ot_g0 =0
        for b in range(B): ot_g0 += compute_pair_ot(g1,misaligned_residual[b,0],reg)
        ot_g0 /= B

        ot_g1 =0
        for b in range(B): ot_g1 += compute_pair_ot(g1,misaligned_residual[b,1],reg)
        ot_g1 /= B

        ot_g2 =0
        for b in range(B): ot_g2 += compute_pair_ot(g1,misaligned_residual[b,2],reg)
        ot_g2 /= B

        print("[0_2: .2%e] [0_1: .2%e] [1_2: .2%e] [g_g: %.2e] [g_0: %.2e] [g_1: %.2e] [g_1: %.2e]" % (ot_02,ot_01,ot_12,ot_gg,ot_g0,ot_g1,ot_g2) )

        """
        compare 
        - iid gaussian and actual residuals [baseline]
        - wrong residual (misaligned - raw_img) v.s. another wrong residual
        - wrong residual (misaligned - raw_img) v.s. middle residual (burst[N//2] - raw_img)

        this looks like its say the noise on the input middle frame is not iid gaussian noise
        """

        # -- compute ot loss to optimize --
        # residuals = aligned - rec_img.unsqueeze(1).repeat(1,N,1,1,1)
        # residuals = rearrange(residuals,'b n c h w -> b n (h w) c')
        # ot_loss = ot_frame_pairwise_bp(residuals,reg=1.0,K=5)
        # ot_coeff = 1 - .997**cfg.global_step

        cfg.global_step += 1
    total_loss /= len(train_loader)
    return total_loss,record

def run_ot_v_displacement(cfg,criterion,train_loader):

    reg = 1.0
    for batch_idx, (burst, residuals, raw_img, directions) in enumerate(train_loader):

        # -- init --
        N,B,C,H,W = burst.shape
        burst = burst.cuda(non_blocking=True)
        raw_img = raw_img.cuda(non_blocking=True) - 0.5
        
        # -- compute residuals --
        misaligned_residual = rearrange(burst - raw_img.unsqueeze(0).repeat(N,1,1,1,1),'n b c h w -> b n (h w) c')

        # -- save img --
        res = burst - raw_img.unsqueeze(0).repeat(N,1,1,1,1)
        res = rearrange(torch.cat([res,raw_img.unsqueeze(0)],dim=0),'n b c h w -> b n c h w')
        burst_r = rearrange(torch.cat([burst,raw_img.unsqueeze(0)],dim=0),'n b c h w -> b n c h w')
        img = rearrange(burst_r,'b n c h w -> (b n) c h w')
        # img = rearrange(torch.cat([burst_r,res],dim=1),'b n c h w -> (b n) c h w')
        fn = f"./test_ot_v_displacement_ex_{batch_idx}.png"
        tv_utils.save_image(img,fn,nrow=B,normalize=True)
        print(f"Wrote images to [{fn}]")
        
        # print("25/255 = {:.2e}".format(25/255))
        # for b in range(B):
        #     for n in range(N):
        #         print("({:d},{:d}): {:.2e} +/- {:.2e}".format(b,n,misaligned_residual[b,n].mean(),misaligned_residual[b,n].std()))

        # -- compute ot for pairwise of misaligned images on sides --
        reg = 0.25
        losses = pd.DataFrame({'d':[],'ot_ave':[],'ot_std':[],'kl_ave':[],'kl_std':[]})
        ot_losses = np.zeros((B,N//2 + N % 2))
        kl_losses = np.zeros((B,N//2 + N % 2))
        for n in range(N//2 + N % 2):
            b_ot_losses,b_kl_losses = [],[]
            for b in range(B):
                #ot_loss = compute_pair_ot(misaligned_residual[b,n],misaligned_residual[b,N//2],reg).item()
                a = misaligned_residual[b,n]
                noise = torch.zeros_like(a)
                # noise = torch.normal(torch.zeros_like(a),std=0.001/255)
                ot_loss = compute_pair_ot(misaligned_residual[b,n],noise,reg).item()
                kl_loss = compute_binned_kl(misaligned_residual[b,n],misaligned_residual[b,N//2]).item()
                ot_losses[b,n] = ot_loss
                kl_losses[b,n] = kl_loss
                b_ot_losses.append(ot_loss),b_kl_losses.append(kl_loss)
            append_loss = {'d':int(cfg.dynamic.ppf*(N//2 - n)),
                           'ot_ave':np.mean(b_ot_losses),'ot_std':np.std(b_ot_losses),
                           'kl_ave':np.mean(b_kl_losses),'kl_std':np.std(b_kl_losses),}
            losses = losses.append(append_loss,ignore_index=True)
        
        print(losses)
        # -- plot losses vs. displacement --
        fig,ax = plt.subplots()
        ax.errorbar(losses['d'],losses['ot_ave'],yerr=losses['ot_std'],label='ot')
        ax.errorbar(losses['d'],losses['kl_ave'],yerr=losses['kl_std'],label='kl')
        ax.set_title("Verifying Impact of Displacement")
        ax.set_xlabel("Number of Pixels")
        ax.set_ylabel("Loss")
        ax = add_legend(ax,'Type',['ot','kl'])
        fn = f"test_ot_v_displacement_plot_{batch_idx}.png"
        plt.savefig(fn,dpi=300)
        plt.close("all")
        print(f"Wrote ot loss image to [{fn}]")
                
        # -- plot top K losses vs displacement --
        K = 20

        b_ot_losses = np.sum(ot_losses,1)
        ot_args = np.argsort(-b_ot_losses)[:K]
        ot_args_bk = np.argsort(b_ot_losses)[:K]        
        ot_frame_ave = np.mean(ot_losses[ot_args,:],0)
        ot_frame_std = np.std(ot_losses[ot_args,:],0)

        b_kl_losses = np.sum(kl_losses,1)
        kl_args = np.argsort(-b_kl_losses)[:K]
        kl_args_bk = np.argsort(b_kl_losses)[:K]
        kl_frame_ave = np.mean(kl_losses[kl_args,:],0)
        kl_frame_std = np.std(kl_losses[kl_args,:],0)

        print(ot_frame_ave.shape)
        print(ot_frame_std.shape)
        print(losses['d'])
        
        fig,ax = plt.subplots()
        ax.errorbar(losses['d'],ot_frame_ave,yerr=ot_frame_std,label='ot')
        ax.errorbar(losses['d'],kl_frame_ave,yerr=kl_frame_std,label='kl')
        ax.set_title("Verifying Impact of Displacement (Top-K)")
        ax.set_xlabel("Number of Pixels")
        ax.set_ylabel("Loss")
        ax = add_legend(ax,'Type',['ot','kl'])
        fn = f"test_ot_v_displacement_plot_topk_{batch_idx}.png"
        plt.savefig(fn,dpi=300)
        plt.close("all")
        print(f"Wrote plot file {fn}")

        # -- show the top K images for ot and kl --

        fn = f"./ot_args_test_displacement_topk_{batch_idx}.png"
        img = rearrange(burst_r[ot_args],'b n c h w -> (b n) c h w')
        tv_utils.save_image(img,fn,nrow=N+1,normalize=True)
        print(f"Wrote example ot images file {fn}")

        fn = f"./ot_args_test_displacement_bottomk_{batch_idx}.png"
        img = rearrange(burst_r[ot_args_bk],'b n c h w -> (b n) c h w')
        tv_utils.save_image(img,fn,nrow=N+1,normalize=True)
        print(f"Wrote example ot images file {fn}")

        fn = f"./kl_args_test_displacement_topk_{batch_idx}.png"
        img = rearrange(burst_r[kl_args],'b n c h w -> (b n) c h w')
        tv_utils.save_image(img,fn,nrow=N+1,normalize=True)
        print(f"Wrote example kl images file {fn}")

        fn = f"./kl_args_test_displacement_bottomk_{batch_idx}.png"
        img = rearrange(burst_r[kl_args_bk],'b n c h w -> (b n) c h w')
        tv_utils.save_image(img,fn,nrow=N+1,normalize=True)
        print(f"Wrote example kl images file {fn}")

        # -- compute ot for pairwise of misaligned images to middle --
        g1 = torch.normal(torch.zeros((64*64*3)),25/255).view(-1,3)
        g2 = torch.normal(torch.zeros((64*64*3)),25/255).view(-1,3)
        g1,g2 = g1.cuda(non_blocking=True),g2.cuda(non_blocking=True)
        ot_gg = compute_pair_ot(g1,g2,reg)

        print("[g_g: %.2e]" % (ot_gg) )

        """
        compare 
        - iid gaussian and actual residuals [baseline]
        - wrong residual (misaligned - raw_img) v.s. another wrong residual
        - wrong residual (misaligned - raw_img) v.s. middle residual (burst[N//2] - raw_img)

        this looks like its say the noise on the input middle frame is not iid gaussian noise
        """

        # -- compute ot loss to optimize --
        # residuals = aligned - rec_img.unsqueeze(1).repeat(1,N,1,1,1)
        # residuals = rearrange(residuals,'b n c h w -> b n (h w) c')
        # ot_loss = ot_frame_pairwise_bp(residuals,reg=1.0,K=5)
        # ot_coeff = 1 - .997**cfg.global_step

        cfg.global_step += 1
    total_loss /= len(train_loader)
    return total_loss,record


def run_test_xbatch(cfg,criterion,train_loader):

    reg = 1.0
    for batch_idx, (burst, residuals, raw_img, directions) in enumerate(train_loader):

        # -- init --
        N,B,C,H,W = burst.shape
        burst = burst.cuda(non_blocking=True)
        raw_img = raw_img.cuda(non_blocking=True) - 0.5
        
        # -- compute residuals --
        misaligned_residual = rearrange(burst - raw_img.unsqueeze(0).repeat(N,1,1,1,1),'n b c h w -> b n (h w) c')

        # -- save img --
        res = burst - raw_img.unsqueeze(0).repeat(N,1,1,1,1)
        res = rearrange(torch.cat([res,raw_img.unsqueeze(0)],dim=0),'n b c h w -> b n c h w')
        burst_r = rearrange(torch.cat([burst,raw_img.unsqueeze(0)],dim=0),'n b c h w -> b n c h w')
        img = rearrange(burst_r,'b n c h w -> (b n) c h w')
        # img = rearrange(torch.cat([burst_r,res],dim=1),'b n c h w -> (b n) c h w')
        fn = f"./test_ot_loss_{batch_idx}.png"
        tv_utils.save_image(img,fn,nrow=B,normalize=True)
        print(f"Wrote images to [{fn}]")
        
        # print("25/255 = {:.2e}".format(25/255))
        # for b in range(B):
        #     for n in range(N):
        #         print("({:d},{:d}): {:.2e} +/- {:.2e}".format(b,n,misaligned_residual[b,n].mean(),misaligned_residual[b,n].std()))

        # -- compute ot for pairwise of misaligned images on sides --
        ot_losses = pd.DataFrame({'b1':[],'b2':[],'n1':[],'n2':[],'loss':[]})
        ot_imgs = []
        for b1 in range(B):
            for b2 in range(B):
                # if b1 > b2: continue
                ot_img = np.zeros((3,N,N))
                for n1 in range(N):
                    for n2 in range(N):
                        ot_loss = compute_pair_ot(misaligned_residual[b1,n1],misaligned_residual[b2,n2],reg).item()
                        append_loss = {'b1':b1,'b2':b2,'n1':n1,'n2':n2,'loss':ot_loss}
                        ot_losses = ot_losses.append(append_loss,ignore_index=True)
                        ot_img[:,n1,n2] = ot_loss
                ot_imgs.append(torch.Tensor(ot_img))
        fn = f"ot_losses_{batch_idx}.png"
        tv_utils.save_image(ot_imgs,fn,nrow=B,normalize=True)
        print(f"Wrote ot loss image to [{fn}]")
                

        ot_losses = ot_losses.astype({'b1': 'int32','b2':'int32','n1':'int32','n2':'int32','loss':'float'})

                
        # -- compute ot for pairwise of misaligned images to middle --
        g1 = torch.normal(torch.zeros((64*64*3)),25/255).view(-1,3)
        g2 = torch.normal(torch.zeros((64*64*3)),25/255).view(-1,3)
        g1,g2 = g1.cuda(non_blocking=True),g2.cuda(non_blocking=True)
        ot_gg = compute_pair_ot(g1,g2,reg)

        print("[g_g: %.2e]" % (ot_gg) )
        print(ot_losses)

        """
        compare 
        - iid gaussian and actual residuals [baseline]
        - wrong residual (misaligned - raw_img) v.s. another wrong residual
        - wrong residual (misaligned - raw_img) v.s. middle residual (burst[N//2] - raw_img)

        this looks like its say the noise on the input middle frame is not iid gaussian noise
        """

        # -- compute ot loss to optimize --
        # residuals = aligned - rec_img.unsqueeze(1).repeat(1,N,1,1,1)
        # residuals = rearrange(residuals,'b n c h w -> b n (h w) c')
        # ot_loss = ot_frame_pairwise_bp(residuals,reg=1.0,K=5)
        # ot_coeff = 1 - .997**cfg.global_step

        cfg.global_step += 1
    total_loss /= len(train_loader)
    return total_loss,record


def ot_frame_pairwise_bp(residuals,reg=0.5,K=3):
    """
    :param residuals: shape [B N D C]
    """
    
    # -- init --
    B,N,D,C = residuals.shape

    # -- create triplets
    S = B*K
    indices,S = create_ot_indices(B,N,S)

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
            loss = sink_stabilized(M,reg).item()
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
        loss = compute_pair_ot(residuals[i],residuals[j],reg)
        ot_loss += loss
        weight = (torch.sum(ri**2) + torch.sum(rj**2)).item()
        ot_loss_w += weight * loss
    return ot_loss,ot_loss_w

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

    ri = (scale*ri).type(torch.IntTensor)
    rj = (scale*rj).type(torch.IntTensor)

    cri = torch.bincount(ri).type(torch.FloatTensor)
    crj = torch.bincount(rj).type(torch.FloatTensor)

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
    
def compute_pair_ot(ri,rj,reg):
    M = torch.sum(torch.pow(ri.unsqueeze(1) - rj,2),dim=-1)
    loss = sink_stabilized(M,reg)
    return loss

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

def create_ot_indices(B,N,S):
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
    r_idx = npr.choice(range(P),S)
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

def write_input_output(cfg,burst,aligned,filters,directions):

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

    afilters = torch.abs(filters)
    sfilters = torch.sum(afilters,dim=2)
    zero = torch.FloatTensor([0.]).to(filters.device)
    wfilters = torch.where(sfilters > float(1.), sfilters, zero)
    return torch.sum(wfilters)
