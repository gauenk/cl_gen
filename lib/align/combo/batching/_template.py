
from abc import ABCMeta,abstractmethod
from easydict import EasyDict as edict


class EvalTemplate(metaclass=ABCMeta):

    def __init__(self,score_fxn,score_fxn_name,patchsize,
                 block_batchsize,noise_info,gpuid=1):
        self.score_fxn = score_fxn
        self.score_fxn_name = score_fxn_name
        self.patchsize = patchsize
        self.block_batchsize = block_batchsize
        self.noise_info = noise_info
        self.indexing = edict({})
        self.samples = edict({'scores':[],'blocks':[]})
        self.gpuid = gpuid
        self.score_cfg = edict()

    def _clear(self):
        self.samples = edict({'scores':[],'blocks':[]})

    def _update_gpuid(self,device):
        gpuid = device.index
        if gpuid != self.gpuid:
            print(f"Updating EvalScores gpuid from {self.gpuid} to {gpuid}")
            self.gpuid = gpuid

    @abstractmethod
    def compute_batch_scores(self):
        pass

    @abstractmethod
    def exec_batch(self):
        pass

    @abstractmethod
    def compute_scores(self):
        pass

    @abstractmethod
    def compute_topK(self):
        pass

    @abstractmethod
    def score_burst_from_flow(self):
        pass

    @abstractmethod
    def score_burst_from_blocks(self):
        pass
    
