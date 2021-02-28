from .dynamic import GlobalCameraMotionTransform

__all__ = ["get_dynamic_transform"]

def get_dynamic_transform(dynamic_info,noise_trans,load_res=False):
    if dynamic['bool'] == False: 
        raise ValueError("We must set dynamics = True for the dynamic dataset loader.")
    if dynamic['mode'] == 'global':
        return GlobalCameraMotionTransform(dynamic,noise_trans,load_res)
    elif dynamic['mode'] == 'local':
        raise NotImplemented("No local motion coded.")
    else:
        raise ValueError("Dynamic model [{dynamic['mode']}] not found.")




