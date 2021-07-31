
from PIL import Image
import flow_utils

class MPI_Sentil():

    def __init__(self,root,noise_params,dynamic_params):
        self.root = root
        self.noise_params = noise_params
        self.dynamic_params = dynamic_params

    def __getitem__(self,index):

        # -- load data to memory --
        img_fns = self.img_files[index]
        flow_fns = self.flow_files[index]
        imgs,flows = [],[]
        for img_fn in img_fns:
            imgs.append(Image.open(img_fn))
        for flow_fn in flow_fns:
            flows.append(Image.open(flow_fn))

        # -- apply noisy, dynamics --
        sample = {'burst':burst,'clean':clean,'flow':flow}
        return sample
