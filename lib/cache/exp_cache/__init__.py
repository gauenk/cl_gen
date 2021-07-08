"""

Read and write experiments based on configs


"""

import shutil,os
from pathlib import Path

from cache_io.uuid_cache import UUIDCache
from cache_io.tensor_cache import TensorCache

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

    def convert_tensors_to_files(self,path,data):
        return self.tensor_cache.convert_tensors_to_files(path.parents[0],data)

    def convert_files_to_tensors(self,path,data):
        return self.tensor_cache.convert_files_to_tensors(path.parents[0],data)
    
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
        for config in tqdm(exps):
            results = self.load_exp(config)
            uuid = self.get_uuid(config)
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
    #   Supporting Functions
    # -------------------------

    def read_results(self,uuid):
        uuid_path = self.root / uuid
        if not uuid_path.exists(): return None
        results_path = uuid_path / "results.pkl"
        results = read_results_file(results_path)
        return results

    def write_results(self,uuid,results):
        uuid_path = self.root / Path(uuid)
        if not uuid_path.exists(): path.mkdir(parents=True)
        results_path = uuid_path / "results.pkl"
        write_results_file(results_path,results)

    def check_results_exists(self,uuid):
        path = self.root / Path(uuid) / "results.pkl"
        return path.exists()

    def write_results_file(results_path,data):
        data_files = self.convert_tensors_to_files(path.parents[0],data)
        with open(path,'w') as f:
            json.dump(data_files,f)

    def read_results_file(results_path):
        with open(path,'r') as f:
            data = json.load(f)
        data_tensors = self.convert_files_to_tensors(path.parents[0],data)
        return data_tensors


