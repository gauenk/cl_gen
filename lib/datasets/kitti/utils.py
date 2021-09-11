import os,re,glob,cv2
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image
from einops import rearrange

VALIDATE_INDICES = dict()
VALIDATE_INDICES['2012'] = [0, 12, 15, 16, 17, 18, 24, 30, 38, 39, 42, 50, 54, 59, 60, 61, 77, 78, 81, 89, 97, 101, 107, 121, 124, 142, 145, 146, 152, 154, 155, 158, 159, 160, 164, 182, 183, 184, 190]
VALIDATE_INDICES['2015'] = [10, 11, 12, 25, 26, 30, 31, 40, 41, 42, 46, 52, 53, 72, 73, 74, 75, 76, 80, 81, 85, 86, 95, 96, 97, 98, 104, 116, 117, 120, 121, 126, 127, 153, 172, 175, 183, 184, 190, 199]

# ======== PLEASE MODIFY ========
kitti_root = r"/srv/disk3tb/home/gauenk/data/kitti/"
VERBOSE = True


def read_frame(path_images,burst_id,fid):
    frame_path = Path(os.path.join(path_images, '%s_%s.png' % (burst_id, fid)))
    if not frame_path.exists():
        raise IndexError(f"Frame {str(frame_path)} does not exist.")
    img = cv2.cvtColor(cv2.imread(str(frame_path)),cv2.COLOR_BGR2RGB) # (h, w, c)
    return img

def read_ishape(path_images,burst_id,fid):
    frame_path = Path(os.path.join(path_images, '%s_%s.png' % (burst_id, fid)))
    w,h = Image.open(frame_path).size
    ishape = (h,w,3)
    return ishape

def vprint(*args,**kwargs):
    if VERBOSE:
        print(*args,**kwargs)

def dir_to_burst_info(path_images):
    # -- get burst ids --
    burst_ids = []
    glob_path = str(Path(path_images) / "*")
    match_str = r"(?P<id>[0-9]{6})_(?P<t>[0-9]+)"
    for full_path in glob.glob(glob_path):
        stem = Path(full_path).stem
        match = re.match(match_str,stem).groupdict()
        group_id = match['id']
        # group_t = match['t']
        burst_ids.append(group_id)
    burst_ids = np.unique(burst_ids)

    # -- get burst for each frame --
    STANDARD_FRAMES = 21
    burst_info = {'ids':[],'nframes':[],'ref_t':[]}
    for burst_id in burst_ids:
        glob_path = str(Path(path_images) / Path("%s_*png" % burst_id))
        burst_info['ids'].append(burst_id)
        burst_info['nframes'].append(len(glob.glob(glob_path)))
        if burst_info['nframes'][-1] == STANDARD_FRAMES:
            burst_info['ref_t'].append(burst_info['nframes'][-1] // 2)
        else:
            nums = []
            for burst_t in glob.glob(glob_path):
                stem = Path(burst_t).stem
                match = re.match(match_str,stem).groupdict()
                group_t = int(match['t'])
                nums.append(group_t)
            nums = sorted(nums)
            T = len(nums)
            ref_t = nums[T//2]
            burst_info['ref_t'].append(ref_t)

    # -- to pandas --
    burst_info = pd.DataFrame(burst_info)
    burst_info = burst_info.set_index("ids")
    return burst_info

