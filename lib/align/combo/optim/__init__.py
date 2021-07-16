"""
Optimizers over the Combinatorial Search Space

"""

# import align.combo.optim
import align.combo.optim.v3 as v3

def get_optim(version):
    if version == "v3":
        # optim.v3
        return None
    else:
        raise ValueError(f"Uknown optimizer version [{version}]")
    

