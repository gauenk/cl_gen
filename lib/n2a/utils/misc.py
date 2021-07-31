
# -- python imports --
from tqdm import tqdm

# -- [local] project imports --
from .io import load_exp,get_uuid
from .mesh import create_mesh,get_setup_fxn

def load_records(version="v2"):
    # -- Load Experiment Mesh --
    experiments,order = create_mesh(version)
    config_setup = get_setup_fxn(version)

    # -- Run Experiment --
    records = []
    for config in tqdm(experiments):
        results = load_exp(config)
        uuid = get_uuid(config)
        if results is None: continue
        append_to_record(records,uuid,config,results)
    return records
    
def append_to_record(records,uuid,config,results):
    record = {'uuid':uuid}
    for key,value in config.items():
        record[key] = value
    for result_id,result in results.items():
        record[result_id] = result
    records.append(record)


def get_ref_block_index(nblocks): return nblocks**2//2 + (nblocks//2)*(nblocks%2==0)
    
