"""
The object for the cell tracking dataset


"""


class CellTracking():

    def __init__(self,root,split,nframes,noise_info,nnf_K,nnf_ps,nnf_exists):
        self.root = root
        paths = get_cell_tracking_paths(root)
        self.paths = paths
        self.split = split
        self.nframes = nframes
        self.noise_info = noise_info
        self.nnf_K = nnf_K
        self.nnf_ps = nnf_ps
        self.nnf_exists = nnf_exists
        self.read_resize = (370, 1224)
        self.dataset = self._read_dataset_paths(paths,parts,nframes,
                                                self.read_resize,nnf_K,
                                                nnf_ps,nnf_exists)


    def _read_dataset_paths(self,paths,parts,nframes,
                            read_resize,nnf_K,nnf_ps,nnf_exists):
        if parts in ["train","val","mixed"]:
            return read_dataset_paths(paths,edition,parts,nframes,
                                      resize=read_resize,
                                      nnf_K = nnf_K, nnf_ps = nnf_ps,
                                      nnf_exists=nnf_exists)
        elif parts in ["test"]:
            return read_dataset_testing(paths,edition,nframes,
                                        resize=read_resize,
                                        nnf_K = nnf_K, nnf_ps = nnf_ps,
                                        nnf_exists=nnf_exists)
        else: raise ValueError(f"[KITTI: read_dataset] Uknown part [{parts}]")


def get_cell_tracking_dataset(cfg,mode):
    root = cfg.dataset.root
    root = Path(root)/Path("cell_tracking_challenge")
    data = edict()
    batch_size = cfg.batch_size
    rtype = 'dict' if cfg.dataset.dict_loader else 'list'
    nnf_K = 3
    nnf_ps = 3
    nnf_exists = return_optional(cfg.dataset,'nnf_exists',True)
    if mode in ["dynamic","denoising"]:
        nframes = cfg.nframes
        noise_info = cfg.noise_params
        data.tr = CellTracking(root,"train",nframes,noise_info,
                               nnf_K,nnf_ps,nnf_exists)
        data.val = CellTracking(root,"val",nframes,noise_info,
                                nnf_K,nnf_ps,nnf_exists)
        data.te = CellTracking(root,"test",nframes,noise_info,
                               nnf_K,nnf_ps,nnf_exists)
    else: raise ValueError(f"Unknown KITTI mode {mode}")

    loader = get_loader(cfg,data,batch_size,mode)
    return data,loader

