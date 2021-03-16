import numpy as np
import os, random, re
import pickle, glob

import PIL
from PIL import Image
from easydict import EasyDict as edict
from pathlib import Path
# from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
# from keras.utils import Sequence
# # For Two Encoder Net
# from keras.models import load_model
# from keras import backend as K
# For faster data generation
from multiprocessing import Process

# -- opencv_transforms --
from opencv_transforms import transforms as ocvT
from opencv_transforms import functional as ocvF


# -- pytorch imports --
from torchvision import utils as tv_utils
from torchvision.transforms import functional as tvF

import torch
from einops import rearrange, repeat, reduce
from .common import get_loader


# Set seeds for RNGs
def random_init(seed):
    random.seed(seed)
    np.random.seed(seed)

# Image/file I/O
def get_filenames(directory):
    filenames = [f for f in os.listdir(directory) if f.endswith(".jpg") or f.endswith(".png")]
    random.shuffle(filenames)
    return filenames

def get_npy_filenames(directory):
    filenames = [f for f in os.listdir(directory) if f.endswith(".npy")]
    #random.shuffle(filenames)
    return filenames
    
def load_images(directory, filenames, multiple=1.0, img_4d=False):
    images = []
    for fname in filenames:
        image = img_to_array(load_img(os.path.join(directory, fname), color_mode="grayscale"))*multiple/255.0
        if img_4d:
            image = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
        images.append(image)
    return images

def save_image(image, directory, filename):
    filename = os.path.join(directory, filename)
    #save_img(filename, image)
    image = (np.clip(image[:,:,0] * 255.0, 0, 255)).astype(np.uint8)
    PIL.Image.fromarray(image, "L").save(filename)

def save_file(obj, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    with open(filename, "wb") as ofile:
        pickle.dump(obj, ofile)

# Image processing utils
def random_crop(image, width, height):
    row = np.random.randint(image.shape[0]-height+1)
    col = np.random.randint(image.shape[1]-width+1)
    return image[row:row+height, col:col+width, ...]

def image_resize(image, size):
    return img_to_array(array_to_img(image).resize(size, resample=PIL.Image.LANCZOS))/255.0

def add_QIS_noise(image, alpha, read_noise, nbits=3):
    pix_max = 2**nbits-1
    frame = np.random.poisson(alpha*image) + read_noise*np.random.randn(*image.shape)
    frame = np.round(frame)
    frame = np.clip(frame, 0, pix_max)
    noisy = frame.astype(np.float32) / alpha
    return noisy



def load_crops(directory, filenames, patch_sz, num_patch, jit=2, J=2, real=False):
    size = num_patch*len(filenames)
    window_sz = (patch_sz + 2*jit) * J
    num_channels = 1 if not real else 8 #TODO
    crops = np.empty([size, window_sz, window_sz, num_channels])
    cnt = 0
    for fname in filenames:
        #print(fname)
        if real:
            image = np.load(os.path.join(directory, fname))
        else:
            image = img_to_array(load_img(os.path.join(directory, fname), color_mode="grayscale"))/255.0
        [height, width, _]    = image.shape
        if height < window_sz or width < window_sz:
            continue
        for i in range(num_patch):
            crops[cnt,:,:,:] = random_crop(image, window_sz, window_sz)
            cnt += 1
    crops = crops[:cnt]

    #shuf = np.arange(cnt)
    #np.random.shuffle(shuf)
    #crops = crops[shuf]

    print("%d instances"%cnt)
    return crops, cnt

def make_burst(image, burst_sz, patch_sz, jit, J, rd_crop=False, real=False):
    stack = []
    static_stack = []
    # Decide burst direction
    if not rd_crop:
        x1, x2 = np.random.randint(2*jit*J+1, size=2)
        xs = np.linspace(x1, x2, num=burst_sz)
        ys = np.linspace(0, 2*jit*J, num=burst_sz)
        if np.random.random() < 0.5:
            xs, ys = ys, xs
    # Generate frames
    for i in range(burst_sz):
        frame = image[..., i:i+1] if real else image # real data
        if rd_crop:
            if i == burst_sz//2:
                frame = frame[jit*J:(jit+patch_sz)*J, jit*J:(jit+patch_sz)*J, ...] # center
            else:
                frame = random_crop(frame, patch_sz*J, patch_sz*J)
        else:
            row, col = int(round(xs[i])), int(round(ys[i]))
            frame = frame[row:row+patch_sz*J, col:col+patch_sz*J, ...]
        frame = image_resize(frame, (patch_sz, patch_sz))
        stack.append(frame[:,:,0])
    # Generate static frames for orcale
    if real:
        for i in range(burst_sz):
            frame = image[..., i:i+1]
            if rd_crop:
                frame = frame[jit*J:(jit+patch_sz)*J, jit*J:(jit+patch_sz)*J, ...] # center
            else:
                row, col = int(round(xs[burst_sz//2])), int(round(ys[burst_sz//2]))
                frame = frame[row:row+patch_sz*J, col:col+patch_sz*J, ...]
            frame = image_resize(frame, (patch_sz, patch_sz))
            static_stack.append(frame[:,:,0])
        return np.transpose(stack, [1,2,0]), np.transpose(static_stack, [1,2,0])
    else:
        frame = image[jit*J:(jit+patch_sz)*J, jit*J:(jit+patch_sz)*J, ...]
        frame = image_resize(frame, (patch_sz, patch_sz))
        #stack += np.random.normal(0.0, 0.1, np.array(stack).shape)
        return np.transpose(stack, [1,2,0]), frame # channel last

def gen_data(crops, patch_sz, burst_sz, jit=2, J=2, rd_crop=False, real=False):
    cnt = len(crops)
    num_channels = 1 if not real else burst_sz
    y = np.empty([cnt, patch_sz, patch_sz, num_channels]) # ground truth
    x = np.empty([cnt, patch_sz, patch_sz, burst_sz]) # noisy burst
    for i in range(cnt):
        stack, static_stack = make_burst(crops[i], burst_sz, patch_sz, jit, J, rd_crop=rd_crop, real=real)
        y[i,:,:,0] = stack[:,:,burst_sz//2] #static_stack
        x[i,:,:,:] = stack
        if real:
            y[i,:,:,:] = static_stack
    return y, x, cnt

# Load synthesized QIS data from images (no noise), or load real QIS data
def load_data(directory, filenames, patch_sz, num_patch, burst_sz, jit=2, J=2, rd_crop=False, real=False):
    crops, _ = load_crops(directory, filenames, patch_sz, num_patch, jit=jit, J=J, real=real)
    return gen_data(crops, patch_sz, burst_sz, jit=jit, J=J, rd_crop=rd_crop, real=real)



def create_foreground_images(directory):
    img_path = directory / Path("./images/")
    label_path = directory / Path("./labels/")
    write_dir =  directory / Path("./foreground/")
    if not write_dir.exists(): write_dir.mkdir(parents=True)

    # -- load label filenames --
    label_fns = {}
    for label_fn in glob.glob(str(label_path / Path("./*regions.txt"))):
        label_id = re.match(".*(?P<id>[0-9]{7}).*",label_fn).groupdict()['id']
        label_fns[label_id] = label_fn
        
    # -- load image filenames --
    img_fns = {}
    for img_fn in glob.glob(str(img_path / Path("./*jpg"))):
        img_id = re.match(".*(?P<id>[0-9]{7}).*",img_fn).groupdict()['id']
        img_fns[img_id] = img_fn

    # -- extract foreground and save --
    viz = False
    sample_ids = list(img_fns.keys())
    for sample_id in sample_ids:
        img_fn = img_fns[sample_id]
        label_fn = label_fns[sample_id]        
        labels = np.loadtxt(label_fn,dtype=np.uint8)
        labels = np.repeat(labels[...,None],3,2)
        image = np.array(Image.open(img_fn))
        fg_img = np.ma.masked_array(image,mask=labels!=7).filled(0)
        # fg_img = labels==7
        write_path = write_dir / Path("{}.png".format(sample_id))
        print(f"Writing filename [{write_path}]")
        fg_img = np.array(Image.fromarray(fg_img,"RGB").convert("RGBA"))
        fg_img[...,-1] = labels[:,:,0]==7
        fg_img = Image.fromarray(fg_img,"RGBA")
        fg_img.save(write_path)
        
        if viz:
            fg_img = torch.tensor(rearrange(fg_img,'h w c -> c h w')).type(torch.float)/255.
            image = torch.tensor(rearrange(image,'h w c -> c h w')).type(torch.float)/255.
            tv_utils.save_image(image,"image.png")
            tv_utils.save_image(fg_img,"foreground.png")


def load_crops_bf(directory, patch_sz, num_patch, jit=2, J=2):
    foreground_filenames = get_filenames(os.path.join(directory, "foreground"))
    background_filenames = get_filenames(os.path.join(directory, "images"))

    size = num_patch*len(foreground_filenames)
    window_sz = patch_sz * J
    foreground_sz = (patch_sz - 2*jit) * J
    crops = []
    cnt = 0
    for fname in foreground_filenames:
        imid = fname[:7]
        foreground = Image.open(os.path.join(directory, "foreground", fname))#.resize((foreground_sz, foreground_sz))
        # print(np.array(foreground).shape)
        fg_img = rearrange(torch.tensor(np.array(foreground)).type(torch.uint8),'h w c -> c h w')
        # fg_img[:3] /= 255.
        # print(fg_img.shape)
        fg_img = tvF.resize(fg_img,(foreground_sz, foreground_sz)).numpy()
        fg_img = rearrange(fg_img,'c h w -> h w c')
        foreground = Image.fromarray(fg_img)
        # print(fg_img.shape)
        # print(fg_img.std(),fg_img.mean(),fg_img.max())
        # tv_utils.save_image(fg_img,"fg_img_alpha.png")
        # tv_utils.save_image(fg_img[:3],"fg_img_noalpha.png")
        # exit()

        #image = img_to_array(load_img(os.path.join(directory, fname), color_mode="grayscale"))/255.0

        for i in range(num_patch):
            # Randomly select background
            background = None
            trial = 0
            while background is None:
                fname_bg = random.choice(background_filenames)
                background = Image.open(os.path.join(directory, "images", fname_bg))
                height, width = background.size
                if height < window_sz or width < window_sz or fname_bg[:7] == imid:
                    background = None
                    trial += 1
                if trial == 10:
                    break

            if background is None:
                continue
            # Crop background
            background = Image.fromarray(random_crop(np.array(background), window_sz, window_sz))
            crops.append((background, foreground))
            cnt += 1

            if cnt > 10: break
    print("%d instances"%cnt)
    return crops, cnt

def make_burst_bf(crop, burst_sz, patch_sz, jit, J, rd_crop=False):
    stack = []
    # Decide burst direction
    if not rd_crop:
        x1, x2 = np.random.randint(2*jit*J+1, size=2)
        xs = np.linspace(x1, x2, num=burst_sz)
        ys = np.linspace(0, 2*jit*J, num=burst_sz)
        if np.random.random() < 0.5:
            xs, ys = ys, xs
    # Generate frames
    angle = random.randrange(15)
    for i in range(burst_sz):
        background, foreground = crop
        if rd_crop:
            if i == burst_sz//2:
                x, y = jit, jit
            else:
                x, y = np.random.randint(2*jit*J+1, size=2)
        else:
            x, y = int(round(xs[i])), int(round(ys[i]))

        foreground = np.array(foreground,dtype=np.uint8)
        if np.max(foreground[:,:,-1]) == 1:
            foreground[:,:,-1] *= 255
        foreground = ocvF.rotate(foreground,360 - angle*i)
        foreground = Image.fromarray(foreground)
        frame = Image.new("RGBA", (patch_sz*J, patch_sz*J), (0,0,0,0))
        frame.paste(background, (0,0))
        frame.paste(foreground, (x,y), mask=foreground)
        frame = frame.resize((patch_sz, patch_sz)).convert("RGB")
        frame = np.array(frame)
        frame = frame / 255.0 - 0.5
        stack.append(frame)
    return stack #np.transpose(stack, [1,2,0]) # channel last

def gen_data_bf(crops, patch_sz, burst_sz, jit=2, J=2, rd_crop=False):
    cnt = len(crops)
    y = np.empty([cnt, patch_sz, patch_sz, 1]) # ground truth
    x = np.empty([cnt, patch_sz, patch_sz, burst_sz]) # noisy burst
    for i in range(cnt):
        stack = make_burst_bf(crops[i], burst_sz, patch_sz, jit, J, rd_crop=rd_crop)
        y[i,:,:,0] = stack[:,:,burst_sz//2]
        x[i,:,:,:] = stack
    return y, x, cnt



def get_eccv2020_dataset(cfg,mode):
    data = edict()
    batch_size = cfg.batch_size
    # create_foreground_images("./data/sun2009/")
    if mode in ['default','dynamic']:
        dynamic_info = edict()
        dynamic_info.num_frames = cfg.N
        data = edict()
        data.tr = ECCV2020(cfg.noise_params,dynamic_info)
        data.val,data.te = data.tr,data.tr
    else: raise ValueError(f"Unknown ECCV2020 mode [{mode}]")
    loader = get_loader(cfg,data,batch_size,mode)
    return data,loader

class ECCV2020():


    # self.noise_info.ntype = "qis"
    # self.noise_info.read_noise = 0.25
    # self.noise_info.alpha = 4

    # self.noise_info.ntype = "g"
    # self.noise_info.std = 25.

    def __init__(self,noise_info,dynamic_info):
        self.noise_info = noise_info
        self.dynamic_info = dynamic_info
        self.directory = Path("./data/sun2009/")
        self.frame_size = 128
        self.jit = 2
        self.J = 2
        num_patch = 2
        self.image_pairs, self.num_burst = load_crops_bf(self.directory, self.frame_size,
                                                         num_patch, jit=self.jit, J=self.J)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self,index):

        # -- create images --
        num_frames = self.dynamic_info.num_frames
        image_pair = self.image_pairs[index]
        burst = make_burst_bf(image_pair, num_frames, self.frame_size, self.jit, self.J, rd_crop=False)
        burst = torch.tensor(np.array(burst)).type(torch.float32)
        burst = rearrange(burst,'n h w c -> n c h w')
        raw_img = burst[num_frames//2] + 0.5
        # tv_utils.save_image(burst,"burst.png",normalize=True)

        # -- add noise --
        if self.noise_info.ntype == "qis":
            burst = add_QIS_noise(burst, self.noise_info['ll']['alpha'], self.noise_info['ll']['read_noise'])
        elif self.noise_info.ntype == "g":
            burst = torch.normal(burst,self.noise_info['g']['stddev']/255.)
            
        # -- compat with package --
        spoof_res,spoof_dir = torch.tensor([0.]),torch.tensor([0.])

        # -- create dict --
        rinfo = {}
        rinfo['burst'] = burst
        rinfo['res'] = spoof_res
        rinfo['clean'] = raw_img        
        rinfo['directions'] = spoof_dir
        return rinfo

"""
# Training data generator for Two Encoder Net that generates different noisy samples in each epoch
class BurstSequence2E(Sequence):
    def __init__(self, directory, patch_sz, num_patch, burst_sz, batch_sz, 
            jit=2, J=2, rd_crop=False, noise=True, alpha=4.0, read_noise=0.25, 
            regen_after=0, renoi_after=0, is_train=False, shift_ckpt=None, noisy_ckpt=None):
        # Store crops and generate data
        filenames = get_filenames(directory)
        self.crops, self.num_burst = load_crops_bf(directory, patch_sz, num_patch, jit=jit, J=J)
        self.y, self.xc, _ = gen_data_bf(self.crops, patch_sz, burst_sz, jit=jit, J=J, rd_crop=rd_crop)
        self.x = add_QIS_noise(self.xc, alpha, read_noise) if noise else self.xc
        # Store param for data generating
        self.patch_sz = patch_sz
        self.burst_sz = burst_sz
        self.jit = jit
        self.J = J
        self.rd_crop = rd_crop
        # Store other param
        self.batch_sz = batch_sz
        self.noise = noise
        self.alpha = alpha
        self.read_noise = read_noise
        self.regen_after = regen_after
        self.renoi_after = renoi_after
        self.is_train = is_train
        # Epoch counter
        self.curr_epoch = 0

        # Store pretrained model functors
        def dummy(y_true, y_pred):
            return K.sum(y_pred)
        def load_functor(model, nlayer=19): #19
            inputs = model.input # input placeholder
            outputs = [layer.output for layer in model.layers][1:nlayer+1] # 1st layer to end of encoder
            return K.function([inputs, K.learning_phase()], outputs)  # evaluation function
        if is_train:
            model = load_model(noisy_ckpt, custom_objects={'l2_loss': dummy, 'psnr_metric': dummy})
            self.noisy_enc = load_functor(model)
            model = load_model(shift_ckpt, custom_objects={'l2_loss': dummy, 'annealed_loss': dummy, 'encoder_loss': dummy, 'psnr_metric': dummy})
            self.shift_enc = load_functor(model,nlayer=23)
        else:
            self.noisy_enc, self.shift_enc = None, None
        

    def __len__(self):
        return int(np.ceil(self.num_burst / float(self.batch_sz)))

    def __getitem__(self, idx):
        y = self.y[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        x = self.x[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        x_shift = self.xc[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        # Generate "correct" codes from pretrained encoders
        if self.is_train:
            x_static = np.repeat(y, self.burst_sz, axis=-1)
            x_noisy = add_QIS_noise(x_static, self.alpha, self.read_noise)
            c1_true = np.zeros((self.batch_sz, self.patch_sz, self.patch_sz, 64))
            #c1_true = self.noisy_enc([x_noisy, 1.])[-1]
            #c2_true = self.shift_enc([x, 1.])[-1]
            c2_true = self.shift_enc([x_shift, 1.])[-1]
        else:
            c1_true = np.zeros((self.batch_sz, self.patch_sz, self.patch_sz, 64)) #(..., 4, 4, 64)
            c2_true = np.zeros((self.batch_sz, self.patch_sz, self.patch_sz, 64)) #(..., 4, 4, 64)
        return x, {"xNet1": c1_true, "xNet2": c2_true, "outputs": y}

    def on_epoch_end(self):
        self.curr_epoch += 1
        if self.jit > 0 and self.regen_after > 0 and self.curr_epoch % self.regen_after == 0:
            self.y, self.xc, _ = gen_data_bf(self.crops, self.patch_sz, self.burst_sz, jit=self.jit, J=self.J, rd_crop=self.rd_crop)
            if not (self.noise and self.renoi_after > 0):
                self.x = self.xc
        if self.noise and self.renoi_after > 0 and self.curr_epoch % self.renoi_after == 0:
            self.x = add_QIS_noise(self.xc, self.alpha, self.read_noise)

    # Get noisy, shift, burst, and ground truth for presentation
    def get_images(self, idx):
        y = self.y[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        x = self.x[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        x_shift = self.xc[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        x_static = np.repeat(y, self.burst_sz, axis=-1)
        x_noisy = add_QIS_noise(x_static, self.alpha, self.read_noise)
        images = {"x_shift": x_shift, "x_noisy": x_noisy, "x_burst": x, "y": y}
        return images




# Training data generator for KPN student that generates different noisy samples in each epoch
class BurstSequenceSubKPN(Sequence):
    def __init__(self, directory, patch_sz, num_patch, burst_sz, batch_sz, 
            jit=2, J=2, rd_crop=False, noise=True, alpha=4.0, read_noise=0.25, 
            regen_after=0, renoi_after=0, is_train=False, kpn_ckpt=None, k_sz=5):
        # Store crops and generate data
        filenames = get_filenames(directory)
        self.crops, self.num_burst = load_crops_bf(directory, patch_sz, num_patch, jit=jit, J=J)
        self.y, self.xc, _ = gen_data_bf(self.crops, patch_sz, burst_sz, jit=jit, J=J, rd_crop=rd_crop)
        self.x = add_QIS_noise(self.xc, alpha, read_noise) if noise else self.xc
        # Store param for data generating
        self.patch_sz = patch_sz
        self.burst_sz = burst_sz
        self.jit = jit
        self.J = J
        self.rd_crop = rd_crop
        # Store other param
        self.batch_sz = batch_sz
        self.noise = noise
        self.alpha = alpha
        self.read_noise = read_noise
        self.regen_after = regen_after
        self.renoi_after = renoi_after
        self.is_train = is_train
        self.k_sz = k_sz
        # Epoch counter
        self.curr_epoch = 0

        # Store pretrained model functors
        def dummy(y_true, y_pred):
            return K.sum(y_pred)
        def load_functor(model, nlayer=38):
            inputs = model.input # input placeholder
            outputs = [layer.output for layer in model.layers][1:nlayer+1] # 1st layer to end of encoder
            return K.function([inputs, K.learning_phase()], outputs)  # evaluation function
        if is_train:
            model = load_model(kpn_ckpt, custom_objects={'annealed_loss': dummy, 'psnr_metric': dummy})
            self.kpn_enc = load_functor(model)
        else:
            self.kpn_enc = None


    def __len__(self):
        return int(np.ceil(self.num_burst / float(self.batch_sz)))

    def __getitem__(self, idx):
        y = self.y[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        x = self.x[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        # Generate "correct" codes from pretrained encoders
        if self.is_train:
            x_shift = self.xc[idx * self.batch_sz:(idx + 1) * self.batch_sz]
            kernels = self.kpn_enc([x_shift, 1.])[-1]
        else:
            kernels = np.zeros((self.batch_sz, self.patch_sz, self.patch_sz, self.k_sz**2*self.burst_sz))
        return x, {"kernels": kernels, "outputs": y}

    def on_epoch_end(self):
        self.curr_epoch += 1
        if self.jit > 0 and self.regen_after > 0 and self.curr_epoch % self.regen_after == 0:
            self.y, self.xc, _ = gen_data_bf(self.crops, self.patch_sz, self.burst_sz, jit=self.jit, J=self.J, rd_crop=self.rd_crop)
            if not (self.noise and self.renoi_after > 0):
                self.x = self.xc
        if self.noise and self.renoi_after > 0 and self.curr_epoch % self.renoi_after == 0:
            self.x = add_QIS_noise(self.xc, self.alpha, self.read_noise)

    # Get noisy, shift, burst, and ground truth for presentation
    def get_images(self, idx):
        y = self.y[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        x = self.x[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        x_shift = self.xc[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        x_static = np.repeat(y, self.burst_sz, axis=-1)
        x_noisy = add_QIS_noise(x_static, self.alpha, self.read_noise)
        images = {"x_shift": x_shift, "x_noisy": x_noisy, "x_burst": x, "y": y}
        return images




# Training data generator that generates different noisy samples in each epoch
class BurstSequence(Sequence):
    def __init__(self, directory, patch_sz, num_patch, burst_sz, batch_sz, 
            jit=2, J=2, rd_crop=False, noise=True, alpha=4.0, read_noise=0.25, 
            regen_after=0, renoi_after=0):
        # Store crops and generate data
        filenames = get_filenames(directory)
        self.crops, self.num_burst = load_crops_bf(directory, patch_sz, num_patch, jit=jit, J=J)
        self.y, self.xc, _ = gen_data_bf(self.crops, patch_sz, burst_sz, jit=jit, J=J, rd_crop=rd_crop)
        self.x = add_QIS_noise(self.xc, alpha, read_noise) if noise else self.xc
        # Store param for data generating
        self.patch_sz = patch_sz
        self.burst_sz = burst_sz
        self.jit = jit
        self.J = J
        self.rd_crop = rd_crop
        # Store other param
        self.batch_sz = batch_sz
        self.noise = noise
        self.alpha = alpha
        self.read_noise = read_noise
        self.regen_after = regen_after
        self.renoi_after = renoi_after
        # Epoch counter
        self.curr_epoch = 0

    def __len__(self):
        return int(np.ceil(self.num_burst / float(self.batch_sz)))

    def __getitem__(self, idx):
        y = self.y[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        x = self.x[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        return x, y

    def on_epoch_end(self):
        self.curr_epoch += 1
        if self.jit > 0 and self.regen_after > 0 and self.curr_epoch % self.regen_after == 0:
            self.y, self.xc, _ = gen_data_bf(self.crops, self.patch_sz, self.burst_sz, jit=self.jit, J=self.J, rd_crop=self.rd_crop)
            if not (self.noise and self.renoi_after > 0):
                self.x = self.xc
        if self.noise and self.renoi_after > 0 and self.curr_epoch % self.renoi_after == 0:
            #self.alpha = random.uniform(0.5, 4.0)
            self.x = add_QIS_noise(self.xc, self.alpha, self.read_noise)


"""

# Only for debugging purpose
if __name__ == "__main__":
    pass
    """
    random_init(42)
    directory = "D:/Datasets/VOC2008/test"
    seq = BurstSequence(directory, 64, 4, 8, 4, jit=2, noise=False)
    x, y = seq[0]
    print(y.shape)
    print(x.shape)
    #print(y[0,:,:,0])
    #print(x[0,:,:,0])
    array_to_img(y[0,:,:,:]).show()
    for i in range(8):
        array_to_img(np.reshape(x[0,:,:,i], (64,64,1))).show()
    """
