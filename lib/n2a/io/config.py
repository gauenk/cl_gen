
# -- python --
from pathlib import Path
from easydict import EasyDict as edict


# -- settings --
from settings import ROOT_PATH
from .utils import print_formatted_exp_info,compare_config


class Config():

    def __init__(self,base,root_version,uuid_version,root_skel,uuid_file_skel):
        self._base = base
        self.root_version = root_version
        self.uuid_version = uuid_version
        self._root_skel = root_skel
        self._uuid_file_skel = uuid_file_skel
        self.verbose = False

    @property
    def root(self):
        filename = self._base / Path(self._root_skel.format(self.root_version))
        return filename

    @property
    def uuid_file(self):
        filename = self.root / Path(self._uuid_file_skel.format(self.uuid_version))
        return filename
    
    @staticmethod
    def compare_config(existing_config,proposed_config):
        for key,value in existing_config.items():
            if proposed_config[key] != value: return False
        return True

    def init_uuid_file(self):
        if verbose: print(f"Init [{self.uuid_file}]")
        if self.uuid_file.exists(): return None    
        data = edict({'uuid':[],'config':[]})
        self.write_uuid_file(data)
    
    def get_uuid_from_config(exp_config):
        data = self.read_uuid_file()
        if data is None:
            self.init_uuid_file()
            return -1
        for uuid,config in zip(data.uuid,data.config):
            match = compare_config(config,exp_config)
            if match: return uuid
        return -1 # no match
    
    def get_config_from_uuid(exp_uuid):
        data = self.read_uuid_file()
        if data is None:
            self.init_uuid_file()
            return -1
        for uuid,config in zip(data.uuid,data.config):
            if uuid == exp_uuid: return config
        return -1 # no match
    
    def get_uuid(exp_config):
        uuid = get_uuid_from_config(exp_config)    
        if uuid == -1:
            if verbose: print("Creating a new UUID and adding to cache file.")
            uuid = str(uuid_gen.uuid4())
            new_pair = edict({'uuid':uuid,'config':exp_config})
            self.append_new_pair(new_pair)
            return uuid
        else:
            if verbose: print(f"Exp Config Already has a UUID {uuid}")
            return uuid
    
    def read_uuid_file(self):
        if self.verbose: print(f"Reading: [{self.uuid_file}]")
        if not self.uuid_file.exists(): return None
        with open(self.uuid_file,'r') as f:
            data = edict(json.load(f))
        return data
    
    def write_uuid_file(self,data):
        if self.verbose: print(f"Writing: [{self.uuid_file}]")
        if not self.uuid_file.parents[0].exists():
            self.uuid_file.parents[0].mkdir(parents=True)
        with open(self.uuid_file,'w') as f:
            json.dump(data,f)
    
    def append_new_field(cfg,version,field,default,new_data):
    
        # -- open stored data --
        default_version = self.version
        data = self.read_uuid_file()
    
        # -- check if key exists & set default --
        configs = pd.DataFrame(data.config)
        keys = list(configs.columns)
        if field in keys:
            if verbose: print(f"Not appending new field [{field}]. Field already exists.")
            return None
        configs[field] = default
        data.config = configs.to_dict()
    
        # -- apply non-default values to data --
        for uuid,new_results in zip(new_data.uuid,new_data.results):
            index = np.where(data.uuid == uuid)[0]
            data.config[index][field] = new_results
    
        # -- write updated uuid file --
        self.version = version
        write_uuid_file(self,data)
        self.version = default_version
        if self.verbose:
            print(f"Wrote new uuid cache with new field. Version [{self.version}]")
    
        return 
    
    def append_new_pair(self,new_pair):
        data = self.read_uuid_file()
        existing_uuid = get_uuid_from_config(new_pair.config)
        existing_config = get_config_from_uuid(new_pair.uuid)
        if existing_uuid == -1 and existing_config == -1:
            data.uuid.append(new_pair.uuid)
            data.config.append(new_pair.config)
            self.write_uuid_file(data)
        else:
            if verbose: print("Not appending data to file since data already exists.")
            if existing_uuid != -1 and verbose:
                print(f"UUID already exists: [{new_pair.uuid}]")
            if existing_config == -1 and verbose:
                print(f"Config already exists")
                print_formatted_exp_info(new_pair.config)
    
    
    def get_config_from_uuid_list(uuids):
        configs = []
        for uuid in uuids:
            configs.append(get_config_from_uuid(uuid))
        return configs
    


version = "1p0"
root_version = version
uuid_version = version
explore_root = Path(ROOT_PATH) / "./output/n2a/"
explore_package = "n2a"

base = explore_root / explore_package
root_skel = "{:s}"
uuid_file_skel = "uuid_database_{:s}.json"
cfg = Config(base,root_version,uuid_version,root_skel,uuid_file_skel)
