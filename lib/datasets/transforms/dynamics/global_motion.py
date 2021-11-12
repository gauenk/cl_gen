
# -- python imports --
import numpy as np
from joblib import Parallel, delayed
from functools import partial
from PIL import Image

# -- pytorch imports --
import torch
from torchvision import transforms as thT
import torchvision.transforms.functional as tvF
import torchvision.utils as tvUtils

# -- project imports --
from pyutils.sobel import create_sobel_filter,apply_sobel_filter
from datasets.common import return_optional

class ScaleZeroMean:

    def __init__(self):
        pass

    def __call__(self,pic):
        return pic - 0.5

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

    def __init__(self,info,noise_trans=None,load_res=False):

        # -- setup parameters --
        self.nframes = info['nframes']
        self.ppf = info['ppf']
        self.frame_size = info['frame_size']
        self.load_res = load_res
        self.sobel_filter = create_sobel_filter()

        # -- manage frame size --
        if isinstance(self.frame_size,int):
            self.frame_size = (self.frame_size,self.frame_size)
        if not(self.frame_size is None):
            if self.frame_size[0] <= 64: self.very_small = True
            else: self.very_small = False
            self.min_frame_size = np.min(self.frame_size)
        else:
            self.very_small = None
            self.min_frame_size = None

        # -- optional --
        self.total_pixels = return_optional(info,'total_pixels',0)
        self.random_eraser_bool = return_optional(info,'random_eraser_bool',False)
        self.reset_seed = return_optional(info,'reset_seed',False)
        self.textured = return_optional(info,'textured',False)

        # -- init vars and consts --
        self.random_eraser = thT.RandomErasing()#scale=(0.40,0.80))
        self.PI = 2*torch.acos(torch.zeros(1)).item() 
        self.szm = thT.Compose([ScaleZeroMean()])
        self.noise_trans = noise_trans

    def __call__(self, pic, tl=None):
        
        if self.nframes == 1:
            pics = self.to_tensor(pic)[None,:]
            clean_target = pics
            seq_flow = torch.zeros((1,2))
            ref_flow = torch.zeros((1,2))
            return pics,None,pics,seq_flow,ref_flow,0
        
        # pics = pic.unsqueeze(0).repeat(cfg.nframes,1,1,1) 
        clean_target = None
        middle_index = self.nframes // 2
        h,w = self._get_pic_hw(pic)
        out_frame_size = self.frame_size            
        # tl_init = tl.clone()
        pic = self.to_tensor(pic)

        # -- compute ppf rate given fixed frames --
        if self.total_pixels > 0:
            tpix = float(self.total_pixels)
            nf = self.nframes
            raw_ppf = tpix / (nf-1) if nf > 1 else tpix
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

        # -- jitter --
        direction = [0,0]
        # tl = torch.IntTensor([self.frame_size[0]//2,self.frame_size[1]//2])
        tl = torch.IntTensor([2*self.ppf,2*self.ppf])
        if self.very_small: tl = self._pick_interesting_tl(pic)

        # -- simple, continuous motion --
        # direction = self.sample_direction()
        # if tl is None: tl = self.init_coordinate(direction,h,w)

        # -- compute pixels per frame and resize image for fractions -- 
        if raw_ppf < 1 and raw_ppf > 0:
            print("WARNING: ppf is less than 1. Weird stuff might happen.")
            h_new,w_new = int(h/raw_ppf)+1,int(w/raw_ppf)+1
            tl = torch.IntTensor([int(x.item()/raw_ppf) for x in tl])
            pic = tvF.resize(pic,(h_new,w_new))
            crop_frame_size = int(self.min_frame_size/raw_ppf)+1
            ppf = 1
        else:
            ppf = raw_ppf
            crop_frame_size = self.min_frame_size
        # print(f"ppf: {ppf}")

        # -- smooth, global motion --
        tl_list = [tl.clone()]
        delta_list = [torch.LongTensor([0,0])]
        nframe_iters = self.nframes-1

        # -- jitter --
        tl_list = []
        delta_list = []
        nframe_iters = self.nframes

        # -- random direction box --
        prng = int(2*ppf+1)
        choices = torch.arange(prng**2)# - prng//2
        order = torch.randperm(prng**2)
        choices = choices[order]
        zidx = torch.where(choices==0)[0]
        tmp = choices[0].item()
        choices[0] = 0
        choices[zidx] = tmp
        choices = choices.reshape(prng,prng)

        for i in range(nframe_iters):

            # -- smooth, global motion --
            # step = (torch.round((i+1) * direction * ppf)).type(torch.int)

            # -- jitter --
            mult = 1.
            direction9 = self.sample_direction_num(9,ppf,choices)
            step = (torch.round(mult * direction9 * ppf)).type(torch.int)

            # -- jitter with no resampling of center frame --
            # direction = self.sample_direction()
            # step = (torch.round(mult * direction * ppf)).type(torch.int)

            # print("[a v.s. b]: {} v.s. {}".format(direction9,direction))
            # print("[a v.s. b]: {} v.s. {}".format(step9,step))
            if i == (self.nframes//2): step = torch.zeros_like(step)

            # -- update with step --
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
        h_new,w_new = self._get_pic_hw(pic)
        tl_mid = tl_list[middle_index]
        t,l = tl_mid[0].item(),tl_mid[1].item()
        target = tvF.resized_crop(pic,t,l,crop_frame_size,crop_frame_size,out_frame_size)
        clean_target = self.to_tensor(target)
        
        # -- create noisy frames -- 
        create_frames = partial(self._crop_image,pic,tl_list,
                                crop_frame_size,out_frame_size)
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
        seq_flow = self._motion_dinit_to_seq_flow(delta_list)
        ref_flow = self._motion_dinit_to_ref_flow(delta_list,self.nframes//2)
        return pics,res,clean_target,seq_flow,ref_flow,tl

    def to_tensor(self,tensor):
        if not torch.is_tensor(tensor):
            if isinstance(tensor,np.ndarray):
                tensor = torch.Tensor(tensor)
            else:
                tensor = tvF.to_tensor(tensor)
            tensor /= tensor.max() # normalize
        return tensor

    def _get_pic_hw(self,pic):
        if torch.is_tensor(pic): c,h,w = pic.shape
        elif isinstance(pic,np.ndarray): c,h,w = pic.shape
        else: w,h = pic.size
        return h,w
        
    def _motion_dinit_to_ref_flow(self,delta,ref_t):
        """
        Compute "ref flow" 
        or 
        flow from a reference image to a target image

        E.g. For T frames 
        flow[t] represents the flow from time ref_t to time t
        """
        T,D = delta.shape
        delta = delta[ref_t] - delta
        nd_delta = delta.numpy() # top-left deltas

        # -- we want flow[i,:] = [dx, dy] --
        ref_flow = np.zeros((T,D))
        ref_flow[:,1] = nd_delta[:,0]
        ref_flow[:,0] = nd_delta[:,1]

        # -- flip yaxis --
        ref_flow[:,1] *= -1

        ref_flow = torch.IntTensor(ref_flow)
        return ref_flow

    def _pick_interesting_tl(self,img):
        """
        We assume jitter motion!
        """

        # -- image info --
        img = self.to_tensor(img)
        c,h,w = img.shape

        # -- where are good edges --
        edges = apply_sobel_filter(img)[0]
        thresh = torch.quantile(edges,.6).item()
        edges = edges > thresh

        # -- what is legal --
        pad = int(self.ppf+1)
        hF,wF = self.frame_size
        interior = torch.zeros_like(edges)
        interior[pad:-(pad+hF),pad:-(pad+wF)] = 1.
        
        # -- pick a t,l from both
        mask = torch.logical_and(edges,interior)
        rows,cols = torch.where(mask)
        nsel = rows.shape[0]
        if nsel == 0: rows,cols = [h//2],[w//2]
        idx = int(torch.rand(1).item() * nsel)
        row,col = rows[idx],cols[idx]

        # -- verify --
        legal_hw = (row+wF+pad) < w and (col+hF+pad) < h
        legal_hw = (wF-pad) >= 0 and (hF-pad) >= 0
        assert legal_hw,"The top-left must be within image shape"

        # -- format output --
        tl = torch.IntTensor([row,col]) # top,left
        
        return tl
        
    def _motion_dinit_to_seq_flow(self,delta):
        """
        Compute "burst flow" or flow between 
        each frame in an ordered sequence

        E.g. For T frames 
        flow[t] represents the flow from time t to time t+1
        
        """
        T,D = delta.shape
        ref_t = T//2
        # print("delta")
        # print(delta)
        # delta = delta[ref_t] - delta
        #print(delta)
        nd_delta = delta.numpy() # top-left deltas
        flow = np.zeros((T-1,D))
        flow[:,1] = np.ediff1d(nd_delta[:,0])
        flow[:,0] = np.ediff1d(nd_delta[:,1])
        # -- we want flow[i,:] = [dx, dy] --
        # -- convert to spatial and flip the dx --
        # flow[:,0] *= -1

        # -- convert to spatial and flip the dx --
        flow[:,1] *= -1

        flow = torch.IntTensor(flow)
        #print(flow)
        return flow
        
    def _check_pic_crop_dims(self,pic,t,l,out_frame_size):
        ch,cw = out_frame_size
        _,ih,iw = pic.shape
        ht,wl = abs(ih - t),abs(iw - l)
        if ht < ch or wl < cw:
            print("WARNING: Image Crop contains zero padding.")
            print("t,l",t,l,out_frame_size,pic.shape)

    def _crop_image(self,pic,tl_list,crop_frame_size,out_frame_size,i):
        tl = tl_list[i]
        # print(torch.norm(tl.type(torch.FloatTensor) - tl_init.type(torch.FloatTensor)))
        t,l = tl[0].item(),tl[1].item()
        # -- resizing clean image results in image blur across (t,l) dynamics --
        # -- notably this blur is consistent so we only see disconnect w/ frame idx 0 --
        # pic_i = tvF.resized_crop(pic,t,l,crop_frame_size,crop_frame_size,out_frame_size)

        # -- to tensor & crop --
        pic = self.to_tensor(pic)
        self._check_pic_crop_dims(pic,t,l,out_frame_size)
        pic_i = tvF.crop(pic,t,l,out_frame_size[0],out_frame_size[1])
        # tvUtils.save_image(pic_i,f"cropped_pic_{i}.png")

        # -- pad to ensure correct size --
        # c,h,w = pic_i.shape
        # pads = max(out_frame_size[0]-h,0),max(out_frame_size[1]-w,0)
        # pic_i = tvF.pad(pic_i,pads)

        # -- apply noise --
        res_i = torch.empty(0)
        if (not self.noise_trans is None):
            noisy_pic_i = self.szm(self.noise_trans(pic_i))
            if self.load_res:
                pic_nmlz = self.szm(pic_i)
                res_i = noisy_pic_i - pic_nmlz
            pic_i = noisy_pic_i
        return pic_i,res_i

    def sample_direction_num(self,ndirs,ppf,choices):
        # -- constants --
        negOnes = torch.FloatTensor([-1.,-1.])
        zeros = torch.FloatTensor([0.,0.])

        # -- handle inputs --
        if self.reset_seed:
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
        rand_int = int(ndirs*torch.rand(1).item())

        nrange = int(2*ppf+1)
        mid = nrange//2
        # h = rand_int % nrange - mid
        # w = rand_int // nrange - mid
        # init = torch.FloatTensor([h,w])

        y,x = torch.where(choices == rand_int)
        y,x = y.item(),x.item()
        init = torch.FloatTensor([y-mid,x-mid])
        # print(choices)
        # print(rand_int)
        # print(init)

        # -- ensure a [0,0] exists: swap (-1,-1) with (0,0) --
        # is_negOnes = torch.all(torch.isclose(init,negOnes))
        # is_zeros = torch.all(torch.isclose(init,zeros))
        # if is_negOnes: init = zeros
        # elif is_zeros: init = negOnes

        # print("init,rand_int: ",init,rand_int)
        return init
        # if rand_int == 0:
        #     return torch.FloatTensor([1.,0.])
        # elif rand_int == 1:
        #     return torch.FloatTensor([0.,0.])
        # elif rand_int == 2:
        #     return torch.FloatTensor([0.,1.])
        # else:
        #     raise ValueError(f"Uknown rand int value [{rand_int}]")

    def sample_direction(self):

        # -- seeds --
        if self.reset_seed:
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
        radius = 1
        rand_int = torch.rand(1)
        theta = rand_int * 2 * self.PI
        
        # -- simplify motion --
        # perm = torch.randperm(4)
        # choices = torch.FloatTensor([0,self.PI/2.,self.PI,3*self.PI/2.])
        # theta = choices[perm[0]]

        direction = torch.FloatTensor([radius * torch.cos(theta),
                                       radius * torch.sin(theta)])
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
            init = [h - self.frame_size[0], 0]# bottom-left to top-left
        elif quandrant == 3:
            # bottom-right  to  top-left
            init = [h - self.frame_size[0], w - self.frame_size[1]]
        elif quandrant == 4:
            init = [0,w - self.frame_size[1]] # top-right to top-left
        else:
            raise ValueError("What happened here?")
        # print(direction,odd,quandrant)

        return torch.IntTensor(init)
