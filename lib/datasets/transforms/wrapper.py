

class TransformWrapper():

    def __init__(self,dataset,noise_params,dynamic_params):
        self.dataset = dataset
        self.noise_params = noise_params
        self.noise_xform = get_noise_xform(noise_params)
        self.dynamic_params = dynamic_params
        self.dynamic_xform = get_dynamic_xform(dynamic_params)

    def __getitem(self,index):
        sample = self.dataset[index]
        self.noise_params
        self.dynamic_params = dynamic_params

        
