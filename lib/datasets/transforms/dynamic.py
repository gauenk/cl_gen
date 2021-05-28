
# -- python imports --
import numpy as np
from joblib import Parallel, delayed
from functools import partial

# -- pytorch imports --
import torch
from torchvision import transforms as thT
import torchvision.transforms.functional as tvF

# -- projects imports --
from .misc import ScaleZeroMean

class GlobalCameraMotionTransform():
    """
    Axis-aligned motion
    
    Global camera motion.

    direction: the vector of where we currently go
    delta: the amount of time spent in a current direction

    -- MISC thinking --

    types of global camera motion
    "wiggle": random jittering around the central location
    - high number of direction changes
    "shift": moving in a fixed direction
    - low number of direction changes
    """

    def __init__(self,dynamic,noise_trans=None,load_res=False):
        self.dynamic = dynamic
        self.load_res = load_res
        self.nframes = dynamic['frames']
        self.ppf = dynamic['ppf']
        self.total_pixels = dynamic['total_pixels']
        self.random_eraser_bool = dynamic['random_eraser']
        self.random_eraser = thT.RandomErasing()#scale=(0.40,0.80))
        self.PI = 2*torch.acos(torch.zeros(1)).item() 
        self.frame_size = self.dynamic.frame_size
        self.img_size = 256
        self.to_tensor = thT.Compose([thT.ToTensor()])
        self.szm = thT.Compose([ScaleZeroMean()])
        self.noise_trans = noise_trans
        self.reset_seed = False
        if "reset_seed" in list(dynamic.keys()):
            self.reset_seed = dynamic.reset_seed

    def __call__(self, pic):
        
        # pics = pic.unsqueeze(0).repeat(cfg.nframes,1,1,1) 
        clean_target = None
        middle_index = self.nframes // 2
        w,h = pic.size
        direction = self.sample_direction()

        tl = self.init_coordinate(direction,h,w)

        out_frame_size = (self.frame_size,self.frame_size)
        # tl_init = tl.clone()

        # -- compute ppf rate given fixed frames --
        if self.total_pixels > 0:
            raw_ppf = float(self.total_pixels) / (self.nframes-1) if self.nframes > 1 else float(self.total_pixels)
        else:
            raw_ppf = self.ppf

        # -- keep blur consistent across higher frame rates --
        downsampling_subpix_movement = False
        if downsampling_subpix_movement:
            max_fr = 6
            interp_key = 1
            raw_ppf_lim = self.total_pixels / (max_fr-1)
            h_new_lim,w_new_lim = int(h*raw_ppf_lim)+1,int(w*raw_ppf_lim)+1
            pic = tvF.resize(pic,(h_new_lim,w_new_lim),interp_key)
            # pic_save = self.to_tensor(pic)
            # tvUtils.save_image(pic_save,"rs-test_shrink.png")

        # pic = tvF.resize(pic,(h_new_lim,w_new_lim),interp_key)
        # pic_save = self.to_tensor(pic)
        # tvUtils.save_image(pic_save,"rs-test_during.png")
        
        # h_new,w_new = int(h/raw_ppf)+1,int(w/raw_ppf)+1
        # print(h_new,w_new)
        # pic = tvF.resize(pic,(h_new,w_new),interp_key)
        # pic_save = self.to_tensor(pic)
        # tvUtils.save_image(pic_save,"rs-test_after.png")
        # exit()

        # -- compute pixels per frame and resize image for fractions -- 
        if raw_ppf < 1 and raw_ppf > 0:
            h_new,w_new = int(h/raw_ppf)+1,int(w/raw_ppf)+1
            tl = torch.IntTensor([int(x.item()/raw_ppf) for x in tl])
            pic = tvF.resize(pic,(h_new,w_new))
            crop_frame_size = int(self.frame_size/raw_ppf)+1
            ppf = 1
        else:
            ppf = raw_ppf
            crop_frame_size = self.frame_size
        # print(f"ppf: {ppf}")

        # -- create list of indices -- 
        tl_list = [tl.clone()]
        delta_list = [torch.LongTensor([0,0])]
        for i in range(self.nframes-1):
            step = (torch.round((i+1) * direction * ppf)).type(torch.int)
            tl_i = tl + step
            delta_list.append(step)
            tl_list.append(tl_i)
        delta_list = torch.stack(delta_list,dim=0)
        # a = tl_list[0].type(torch.float)
        # b = tl_list[-1].type(torch.float)
        # print("pix diff",torch.sqrt(torch.sum(( a - b)**2)).item())
        # print(tl_list[0],tl_list[-1],d,w,h,pic.size)

        # print("d",torch.LongTensor([np.array(tl) for tl in tl_list]))
        # -- get clean image --
        w_new,h_new = pic.size
        tl_mid = tl_list[middle_index]
        t,l = tl_mid[0].item(),tl_mid[1].item()
        target = tvF.resized_crop(pic,t,l,crop_frame_size,crop_frame_size,out_frame_size)
        clean_target = self.to_tensor(target)
        
        # -- create noisy frames -- 
        create_frames = partial(self._crop_image,pic,tl_list,crop_frame_size,out_frame_size)
        if self.nframes <= 300:
            pics = []
            res = []
            for i in range(self.nframes):
                pic_i,res_i = create_frames(i)
                pics.append(pic_i),res.append(res_i)
        # print(torch.norm(tl.type(torch.FloatTensor) - tl_init.type(torch.FloatTensor)))
        else:
            nj = np.min([self.nframes // 5,8])
            both = Parallel(n_jobs=nj)(delayed(create_frames)(i) for i in range(self.nframes))
            pics = [x[0] for x in both]
            res = [x[1] for x in both]            
        pics = torch.stack(pics)
        res = torch.stack(res)
        # print(clean_target.min(),clean_target.max(),clean_target.mean())
        if self.random_eraser_bool: pics[middle_index] = self.random_eraser(pics[middle_index])
        flow = self._motion_dinit_to_flow(delta_list)
        return pics,res,clean_target,flow

    def _motion_dinit_to_flow(self,delta):
        nd_delta = delta.numpy()
        T,D = nd_delta.shape
        flow = np.zeros((T-1,D))
        flow[:,1] = np.ediff1d(nd_delta[:,0])
        flow[:,0] = np.ediff1d(nd_delta[:,1])
        # -- we want flow[i,:] = [dx, dy] --
        # -- convert to spatial and flip the dx --
        flow[:,0] *= -1
        flow = torch.IntTensor(flow)
        return flow

        
    def _crop_image(self,pic,tl_list,crop_frame_size,out_frame_size,i):
        tl = tl_list[i]
        # print(torch.norm(tl.type(torch.FloatTensor) - tl_init.type(torch.FloatTensor)))
        t,l = tl[0].item(),tl[1].item()
        # -- resizing clean image results in image blur across (t,l) dynamics --
        # -- notably this blur is consistent so we only see disconnect w/ frame idx 0 --
        # pic_i = tvF.resized_crop(pic,t,l,crop_frame_size,crop_frame_size,out_frame_size)
        pic_i = tvF.crop(pic,t,l,out_frame_size[0],out_frame_size[1])
        res_i = torch.empty(0)
        if (not self.noise_trans is None):
            noisy_pic_i = self.szm(self.noise_trans(self.to_tensor(pic_i)))
            if self.load_res:
                pic_nmlz = self.szm(self.to_tensor(pic_i))
                res_i = noisy_pic_i - pic_nmlz
            pic_i = noisy_pic_i
        else:
            pic_i = self.szm(self.to_tensor(pic_i))
        return pic_i,res_i

    def sample_direction(self):
        if self.reset_seed:
            torch.manual_seed(0)
        radius = 1
        rand_int = torch.rand(1)
        theta = rand_int * 2 * self.PI
        
        # -- simplify motion --
        # perm = torch.randperm(4)
        # choices = torch.FloatTensor([0,self.PI/2.,self.PI,3*self.PI/2.])
        # theta = choices[perm[0]]

        direction = torch.FloatTensor([radius * torch.cos(theta), radius * torch.sin(theta)])
        # print("Axis-aligned dynamics.")
        # direction = torch.FloatTensor([1.,0.])
        return direction

    def init_coordinate(self,direction,h,w):
        
        odd = torch.prod(direction).item() > 0
        quandrant = 0
        if odd:
            if torch.all(direction > 0):
                quandrant = 1
            else:
                quandrant = 3
        elif not odd:
            if direction[1] > 0:
                quandrant = 2
            else:
                quandrant = 4

        init = [-1,-1] # top-left corner
        if quandrant == 1:
            init = [0,0]
        elif quandrant == 2:
            init = [h - self.frame_size, 0]# bottom-left to top-left
        elif quandrant == 3:
            init = [h - self.frame_size, w - self.frame_size] # bottom-right to top-left
        elif quandrant == 4:
            init = [0,w - self.frame_size] # top-right to top-left
        else:
            raise ValueError("What happened here?")
        # print(direction,odd,quandrant)

        return torch.IntTensor(init)
