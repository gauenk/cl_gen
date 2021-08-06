"""

Read and write experiments based on configs


"""

import shutil,os,json,tqdm
import pandas as pd
from pathlib import Path

import numpy as np
from einops import rearrange,repeat

from cache_io.uuid_cache import UUIDCache
from cache_io.tensor_cache import TensorCache

VERBOSE=False

class ExpCache():

    def __init__(self,root,version):
        self.root = root
        self.tensor_cache = TensorCache(root)
        self.uuid_cache = UUIDCache(root,version)
        
    @property
    def version(self):
        return self.uuid_cache.version

    @property
    def uuid_file(self):
        return self.uuid_cache.uuid_file

    def get_uuid_from_config(self,config):
        return self.uuid_cache.get_uuid_from_config(config)

    def convert_tensors_to_files(self,uuid,data):
        return self.tensor_cache.convert_tensors_to_files(uuid,data)

    def convert_files_to_tensors(self,uuid,data):
        return self.tensor_cache.convert_files_to_tensors(uuid,data)
    
    # ---------------------
    #   Primary Functions
    # ---------------------

    def load_exp(self,config):
        uuid = self.get_uuid_from_config(config)
        if uuid == -1: return None
        results = self.read_results(uuid)
        return results

    def save_exp(self,uuid,config,results,overwrite=False):
        check_uuid = self.get_uuid_from_config(config)
        assert check_uuid == -1 or uuid == check_uuid, "Only one uuid per config." 
        exists = self.check_results_exists(uuid)
        if overwrite is True or exists is False:
            if (exists is True) and VERBOSE:
                print("Overwriting Old UUID.")
            if VERBOSE: print(f"UUID [{uuid}]")
            self.write_results(uuid,results)
        else:
            print(f"WARNING: Not writing. UUID [{uuid}] exists.")

    def get_uuid(self,config):
        return self.uuid_cache.get_uuid(config)

    def load_records(self,exps):
        records = []
        for config in tqdm.tqdm(exps):
            results = self.load_exp(config)
            uuid = self.get_uuid(config)
            if results is None: continue
            self.append_to_record(records,uuid,config,results)
        records = pd.DataFrame(records)
        return records

    def append_to_record(self,records,uuid,config,results):
        record = {'uuid':uuid}
        for key,value in config.items():
            record[key] = value
        for result_id,result in results.items():
            record[result_id] = result
        records.append(record)

    def load_flat_records(self,exps):
        """
        Load records but flatten exp configs against
        experiments. Requires "results" to be have 
        equal number of rows.
        """
        records = []
        for config in tqdm.tqdm(exps):
            results = self.load_exp(config)
            uuid = self.get_uuid(config)
            if results is None: continue
            self.append_to_flat_record(records,uuid,config,results)
        records = pd.concat(records)
        return records

    def append_to_flat_record(self,records,uuid,config,results):

        # -- init --
        record = {}

        # -- append results --
        rlen = -1
        for result_id,result in results.items():
            record[result_id] = list(result)
            rlen = len(result)

        # -- repeat uuid --
        uuid_np = repeat(np.array([str(uuid)]),'1 -> r',r=rlen)
        record['uuid'] = list(uuid_np)

        # -- standard append --
        for key,value in config.items():
            record[key] = list(repeat(np.array([value]),'1 -> r',r=rlen))

        # -- create id along axis --
        pdid = np.arange(rlen)
        record['pdid'] = np.arange(rlen)

        # -- repeat config info along result axis --
        # print("All should be equal length.")
        # for key,val in record.items():
        #     print(key,len(val))

        # record = pd.DataFrame().append(record,ignore_index=True)
        record = pd.DataFrame(record,index=pdid)
        records.append(record)

    # -------------------------
    #     Clear Function
    # -------------------------

    def clear(self):
        print("Clearing Cache.")
        uuid_file = self.uuid_file
        if not uuid_file.exists(): return
    
        # -- remove all experiment results --
        data = self.uuid_cache.data
        for uuid in data.uuid:
            uuid_path = self.root / Path(uuid)
            if not uuid_path.exists(): continue
            shutil.rmtree(uuid_path)
            assert not uuid_path.exists(),f"exp cache [{uuid_path}] should be removed."

        # -- remove uuid cache --
        if uuid_file.exists(): os.remove(uuid_file)
        assert not uuid_file.exists(),f"uuid file [{uuid_file}] should be removed."
        self.uuid_cache.init_uuid_file()
    
    # -------------------------
    #   Read/Write Functions
    # -------------------------

    def read_results(self,uuid):
        uuid_path = self.root / uuid
        if not uuid_path.exists(): return None
        results_path = uuid_path / "results.pkl"
        results = self.read_results_file(results_path,uuid)
        return results

    def write_results(self,uuid,results):
        uuid_path = self.root / Path(uuid)
        if not uuid_path.exists(): uuid_path.mkdir(parents=True)
        results_path = uuid_path / "results.pkl"
        self.write_results_file(results_path,uuid,results)

    def check_results_exists(self,uuid):
        path = self.root / Path(uuid) / "results.pkl"
        return path.exists()
    
    def write_results_file(self,path,uuid,data):
        data_files = self.convert_tensors_to_files(uuid,data)
        with open(path,'w') as f:
            json.dump(data_files,f)

    def read_results_file(self,path,uuid):
        with open(path,'r') as f:
            data = json.load(f)
        data_tensors = self.convert_files_to_tensors(uuid,data)
        return data_tensors
