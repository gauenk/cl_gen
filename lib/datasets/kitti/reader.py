import os
import cv2
import numpy as np

VALIDATE_INDICES = dict()
VALIDATE_INDICES['2012'] = [0, 12, 15, 16, 17, 18, 24, 30, 38, 39, 42, 50, 54, 59, 60, 61, 77, 78, 81, 89, 97, 101, 107, 121, 124, 142, 145, 146, 152, 154, 155, 158, 159, 160, 164, 182, 183, 184, 190]
VALIDATE_INDICES['2015'] = [10, 11, 12, 25, 26, 30, 31, 40, 41, 42, 46, 52, 53, 72, 73, 74, 75, 76, 80, 81, 85, 86, 95, 96, 97, 98, 104, 116, 117, 120, 121, 126, 127, 153, 172, 175, 183, 184, 190, 199]

# ======== PLEASE MODIFY ========
kitti_root = r"/srv/disk3tb/home/gauenk/data/kitti/"


def read_dataset(path = None, editions = 'mixed', parts = 'mixed', crop = None, resize = None, samples = None):
    if path is None:
        path = kitti_path

    dataset = dict()
    dataset['image_0'] = []
    dataset['image_1'] = []
    dataset['flow'] = []
    dataset['occ'] = []
    dataset['nnf'] = []
    editions = ('2012', '2015') if editions == 'mixed' else (editions, )

    rs_nnf = 'default' if resize is None else '{:d}_{:d}'.format(*resize)
    for edition in editions:
        path_images = path[edition + 'image']
        path_flows = path[edition + 'flow_occ']
        path_nnf = path[edition + 'nnf']
        num_files = len(os.listdir(path_flows)) - 1
        ind_valids = VALIDATE_INDICES[edition]
        num_valids = len(ind_valids)
        if samples is not None:
            num_files = min(num_files, samples)
        ind = 0
        for k in range(num_files):
            if ind < num_valids and ind_valids[ind] == k:
                ind += 1
                if parts == 'train':
                    continue
            elif parts == 'valid':
                    continue
            img0 = cv2.imread(os.path.join(path_images, '%06d_10.png' % k))
            img1 = cv2.imread(os.path.join(path_images, '%06d_11.png' % k))
            nnf = cv2.imread(os.path.join(path_nnf, rs_nnf, '%06d_10.png' % k), -1)
            flow_occ = cv2.imread(os.path.join(path_flows, '%06d_10.png' % k), -1)
            if crop is not None:
                img0 = img0[-crop[0]:, :crop[1]]
                img1 = img1[-crop[0]:, :crop[1]]
                nnf = nnf[-crop[0]:, :crop[1]]
                flow_occ = flow_occ[-crop[0]:, :crop[1]]
            flow = np.flip(flow_occ[..., 1:3], axis=-1).astype(np.float32)
            flow = (flow - 32768.) / (64.)
            occ = flow_occ[..., 0:1].astype(np.uint8)

            if resize is not None:
                img0 = cv2.resize(img0, resize)
                img1 = cv2.resize(img1, resize)
                flow = cv2.resize(flow, resize) * ((np.array(resize, dtype = np.float32) - 1.0) / (
                    np.array([flow.shape[d] for d in (1, 0)], dtype = np.float32) - 1.0))[np.newaxis, np.newaxis, :]
                occ = cv2.resize(occ.astype(np.float32), resize)[..., np.newaxis]
                flow = flow / (occ + (occ == 0))
                occ = (occ * 255).astype(np.uint8)
            else:
                occ = occ * 255

            dataset['image_0'].append(img0)
            dataset['image_1'].append(img1)
            dataset['flow'].append(flow)
            dataset['occ'].append(occ)
            dataset['nnf'].append(nnf)

    return dataset

def read_dataset_testing(path = None, editions = 'mixed', resize = None, samples = None):
    if path is None:
        path = kitti_path

    dataset = dict()
    dataset['2012'] = dict()
    dataset['2012']['image_0'] = []
    dataset['2012']['image_1'] = []
    dataset['2012']['nnf'] = []
    dataset['2015'] = dict()
    dataset['2015']['image_0'] = []
    dataset['2015']['image_1'] = []
    dataset['2015']['nnf'] = []
    editions = ('2012', '2015') if editions == 'mixed' else (editions, )
    rs_nnf = 'default' if resize is None else '{:d}_{:d}'.format(*resize)
    for edition in editions:
        path_image = path[edition + 'image' + 'test']
        path_nnf = path[edition + 'nnf' + 'test']
        num_files = (len(os.listdir(path_image)) - 10) // 21
        if samples is not None:
            num_files = min(num_files, samples)
        for k in range(num_files):
            img0 = cv2.imread(os.path.join(path_image, '%06d_10.png' % k))
            img1 = cv2.imread(os.path.join(path_image, '%06d_11.png' % k))
            nnf = cv2.imread(os.path.join(path_nnf, rs_nnf, '%06d_10.png' % k))
            if resize is not None:
                img0 = cv2.resize(img0, resize)
                img1 = cv2.resize(img1, resize)
            dataset[edition]['image_0'].append(img0)
            dataset[edition]['image_1'].append(img1)
            dataset[edition]['nnf'].append(nnf)

    return dataset


if __name__ == '__main__':
    dataset = read_dataset(resize = (1024, 436))
    # print(dataset['occ'][0].shape)
