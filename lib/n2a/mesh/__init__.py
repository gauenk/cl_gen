from .v1 import create_mesh as create_mesh_v1
from .v1 import config_setup as config_setup_v1

def create_mesh(version):
    if version == "1":
        return create_mesh_v1()
    else:
        raise ValueError(f"Uknown mesh version {version}.")
    
def get_setup_fxn(version):
    if version == "1":
        return config_setup_v1
    else:
        raise ValueError(f"Uknown mesh version {version}.")

