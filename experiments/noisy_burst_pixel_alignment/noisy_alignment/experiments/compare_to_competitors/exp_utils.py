# -- python imports --
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torchvision.transforms.functional as tvF

# -- project imports --
from align import compute_epe,compute_aligned_psnr,compute_pair_flow_acc

def print_dict_ndarray_0_midpix(dict_ndarray,mid_pix):
    print("-"*50)
    for name,ndarray in dict_ndarray.items():
        print(name,ndarray[0,mid_pix])
        
def remove_center_frame(frames):
    nframes = frames.shape[0]
    nc_frames =torch.cat([frames[:nframes//2],frames[nframes//2+1:]],dim=0)
    return nc_frames

def center_crop_frames(frames,csize=30):
    csize = 30
    cc_frames = edict()
    for name,burst in frames.items():
        cc_frames[name] = tvF.center_crop(aligned_of,(csize,csize))
    return cc_frames
    
def compute_nnf_acc(flows):
    return compute_acc_wrt_ref(flows,"nnf")

def compute_acc_wrt_ref(flows,ref):
    skip_fields = ["clean"]
    accs = edict()
    for field in flows.keys():
        if field in skip_fields: continue
        accs[field] = compute_pair_flow_acc(flows[field],flows[ref])
    return accs

def compute_frames_psnr(frames,isize):
    skip_fields = ["clean"]
    psnrs = edict()
    for field in frames.keys():
        if field in skip_fields: continue
        psnrs[field] = compute_aligned_psnr(frames[field],frames.clean,isize)
    return psnrs

def compute_flows_epe_wrt_ref(flows,ref):
    skip_fields = []
    epes = edict()
    for field in flows.keys():
        if field in skip_fields: continue
        print(field,ref,flows[field].shape,flows[ref].shape)
        epes[field] = compute_epe(flows[field],flows[ref])
    return epes

# def compute_flows_epe(flows):
#     epes = compute_flows_epe_wrt_ref(flows,ref)
    # epes = edict()
    # for field in flows.keys():
    #     epes[field] = compute_epe(flows[field],flows[ref])
    # epes.of = compute_epe(flows.gt,flows.gt)
    # epes.nnf = compute_epe(flows.nnf,flows.gt)
    # epes.split = compute_epe(flows.split,flows.gt)
    # epes.ave_simp = compute_epe(flows.ave_simp,flows.gt)
    # epes.ave = compute_epe(flows.ave,flows.gt)
    # epes.est = compute_epe(flows.est,flows.gt)
    # return epes

def remove_frame_centers(frames):
    nc_frames = edict()
    for name,burst in frames.items():
        nc_frames[name] = remove_center_frame(burst)
    return nc_frames

def print_runtimes(runtimes):
    print("-"*50)
    print("Compute Time [smaller is better]")
    print("-"*50)
    print("[NNF]: %2.3e" % runtimes.nnf)
    print("[Split]: %2.3e" % runtimes.split)
    print("[Ave [Simple]]: %2.3e" % runtimes.ave_simp)
    print("[Ave]: %2.3e" % runtimes.ave)
    print("[Proposed]: %2.3e" % runtimes.est)
    print("[NVOF]: %2.3e" % runtimes.nvof)
    print("[FlowNetv2]: %2.3e" % runtimes.flownet)

def print_verbose_epes(epes_of,epes_nnf):
    print("-"*50)
    print("EPE Errors [smaller is better]")
    print("-"*50)

    print("NNF v.s. Optical Flow.")
    print(epes_of.nnf)
    print("Split v.s. Optical Flow.")
    print(epes_of.split)
    print("Ave [Simple] v.s. Optical Flow.")
    print(epes_of.ave_simp)
    print("Ave v.s. Optical Flow.")
    print(epes_of.ave)
    print("Proposed v.s. Optical Flow.")
    print(epes_of.est)
    print("NVOF v.s. Optical Flow.")
    print(epes_of.nvof)
    print("FlowNetv2 v.s. Optical Flow.")
    print(epes_of.flownet)

    print("Split v.s. NNF")
    print(epes_nnf.split)
    print("Ave [Simple] v.s. NNF")
    print(epes_nnf.ave_simp)
    print("Ave v.s. NNF")
    print(epes_nnf.ave)
    print("Proposed v.s. NNF")
    print(epes_nnf.est)
    print("NVOF v.s. NNF")
    print(epes_nnf.nvof)
    print("FlowNetv2 v.s. NNF")
    print(epes_nnf.flownet)

def print_summary_epes(epes_of,epes_nnf):
    print("-"*50)
    print("Summary of EPE Errors [smaller is better]")
    print("-"*50)
    print("[NNF v.s. Optical Flow]: %2.3f" % epes_of.nnf.mean().item())
    print("[Split v.s. Optical Flow]: %2.3f" % epes_of.split.mean().item())
    print("[Ave [Simple] v.s. Optical Flow]: %2.3f" % epes_of.ave_simp.mean().item())
    print("[Ave v.s. Optical Flow]: %2.3f" % epes_of.ave.mean().item())
    print("[Proposed v.s. Optical Flow]: %2.3f" % epes_of.est.mean().item())
    print("[NVOF v.s. Optical Flow]: %2.3f" % epes_of.nvof.mean().item())
    print("[Split v.s. NNF]: %2.3f" % epes_nnf.split.mean().item())
    print("[Ave [Simple] v.s. NNF]: %2.3f" % epes_nnf.ave_simp.mean().item())
    print("[Ave v.s. NNF]: %2.3f" % epes_nnf.ave.mean().item())
    print("[Proposed v.s. NNF]: %2.3f" % epes_nnf.est.mean().item())
    print("[NVOF v.s. NNF]: %2.3f" % epes_nnf.nvof.mean().item())
    print("[FlowNetv2 v.s. NNF]: %2.ef" % epes_nnf.flownet.mean().item())


def print_verbose_psnrs(psnrs):
    print("-"*50)
    print("PSNR Values [bigger is better]")
    print("-"*50)

    print("Optical Flow [groundtruth v1]")
    print(psnrs.of)
    print("NNF [groundtruth v2]")
    print(psnrs.nnf)
    print("Split [old method]")
    print(psnrs.split)
    print("Ave [simple; old method]")
    print(psnrs.ave_simp)
    print("Ave [old method]")
    print(psnrs.ave)
    print("Proposed [new method]")
    print(psnrs.est)
    print("NVOF")
    print(psnrs.nvof)
    print("FlowNetv2")
    print(psnrs.flownet)

def print_delta_summary_psnrs(psnrs):
    print("-"*50)
    print("PSNR Comparisons [smaller is better]")
    print("-"*50)

    delta_split = psnrs.nnf - psnrs.split
    delta_ave_simp = psnrs.nnf - psnrs.ave_simp
    delta_ave = psnrs.nnf - psnrs.ave
    delta_est = psnrs.nnf - psnrs.est
    delta_nvof = psnrs.nnf - psnrs.nvof
    delta_flownet = psnrs.nnf - psnrs.flownet
    print("ave([NNF] - [Split]): %2.3f" % delta_split.mean().item())
    print("ave([NNF] - [Ave [Simple]]): %2.3f" % delta_ave_simp.mean().item())
    print("ave([NNF] - [Ave]): %2.3f" % delta_ave.mean().item())
    print("ave([NNF] - [Proposed]): %2.3f" % delta_est.mean().item())
    print("ave([NNF] - [NVOF]): %2.3f" % delta_nvof.mean().item())
    print("ave([NNF] - [FlowNet]): %2.3f" % delta_flownet.mean().item())

def print_summary_psnrs(psnrs):
    print("-"*50)
    print("Summary PSNR Values [bigger is better]")
    print("-"*50)

    print("[Optical Flow]: %2.3f" % psnrs.of.mean().item())
    print("[NNF]: %2.3f" % psnrs.nnf.mean().item())
    print("[Split]: %2.3f" % psnrs.split.mean().item())
    print("[Ave [Simple]]: %2.3f" % psnrs.ave_simp.mean().item())
    print("[Ave]: %2.3f" % psnrs.ave.mean().item())
    print("[Proposed]: %2.3f" % psnrs.est.mean().item())
    print("[NVOF]: %2.3f" % psnrs.nvof.mean().item())
    print("[FlowNet]: %2.3f" % psnrs.flownet.mean().item())

def print_nnf_acc(nnf_acc):
    print("-"*50)
    print("NNF Accuracy [bigger is better]")
    print("-"*50)

    print("Split v.s. NNF")
    print(nnf_acc.split)
    print("Ave [Simple] v.s. NNF")
    print(nnf_acc.ave_simp)
    print("Ave v.s. NNF")
    print(nnf_acc.ave)
    print("Proposed v.s. NNF")
    print(nnf_acc.est)
    print("Proposed v.s. NNF")
    print(nnf_acc.nvof)
    print("Proposed v.s. FlowNet")
    print(nnf_acc.flownet)

def print_nnf_local_acc(nnf_acc):
    print("-"*50)
    print("Local NNF Accuracy [bigger is better]")
    print("-"*50)

    print("Split v.s. NNF")
    print(nnf_acc.split)
    print("Ave [Simple] v.s. NNF")
    print(nnf_acc.ave_simp)
    print("Ave v.s. NNF")
    print(nnf_acc.ave)
    print("Proposed v.s. NNF")
    print(nnf_acc.est)
    print("NVOF v.s. NNF")
    print(nnf_acc.nvof)
    print("FLOWNET v.s. NNF")
    print(nnf_acc.flownet)

def is_converted(sample,translate):
    for key1,key2 in translate.items():
        if not(key2 in sample): return False
    return True

def convert_keys(sample):

    translate = {'noisy':'dyn_noisy',
                 'burst':'dyn_clean',
                 'snoisy':'static_noisy',
                 'sburst':'static_clean',
                 'ref_flow':'flow_gt',
                 'seq_flow':'seq_flow',
                 'index':'image_index'}

    if is_converted(sample,translate): return sample
    for field1,field2 in translate.items():
        if not(field1 in sample): continue
        sample[field2] = sample[field1]
        if field2 != field1: del sample[field1]
    return sample



