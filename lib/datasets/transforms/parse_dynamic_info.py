from .dynamics import GlobalCameraMotionTransform,LocalCameraMotionTransform

__all__ = ["get_dynamic_transform"]

def get_dynamic_transform(dynamic_info,noise_trans,load_res=False):
    if 'bool' in dynamic_info.keys():
        if dynamic_info['bool'] == False: 
            raise ValueError("We must set dynamics = True for the dynamic dataset loader.")
    if dynamic_info['mode'] == 'global':
        return GlobalCameraMotionTransform(dynamic_info,noise_trans,load_res)
    elif dynamic_info['mode'] == 'local':
        return LocalCameraMotionTransform(dynamic_info,noise_trans,load_res)
    else:
        raise ValueError("Dynamic model [{dynamic_info['mode']}] not found.")

