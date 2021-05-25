from .eval_mesh import create_eval_mesh
from .coupling_mesh import create_coupling_mesh
from .v1 import create_mesh as create_mesh_v1
from .v2 import create_mesh as create_mesh_v2
from .v2 import config_setup as config_setup_v2
from .v3 import create_mesh as create_mesh_v3
from .v3 import config_setup as config_setup_v3

def create_mesh(name):
    if name == "eval":
        return create_eval_mesh()
    elif name == "coupling":
        return create_coupling_mesh() 
    elif name == "v1":
        return create_mesh_v1()
    elif name == "v2":
        return create_mesh_v2()
    elif name == "v3":
        return create_mesh_v3()
    else:
        raise ValueError(f"Uknown mesh name {name}.")
    

def get_setup_fxn(name):
    if name == "eval":
        return None
    elif name == "coupling":
        return None
    elif name == "v1":
        return None
    elif name == "v2":
        return config_setup_v2
    elif name == "v3":
        return config_setup_v3
    else:
        raise ValueError(f"Uknown mesh name {name}.")

