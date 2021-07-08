import json,pprint

from ._write import *
from ._read import *
from ._utils import *
from ._append import *
from ._debug import *

class UUIDCache():

    def __init__(self,root,version):
        self.root = root
        self.version = version
        self.uuid_file_skel = "uuid_database_{:s}.json"
        self.init_uuid_file()

    @property
    def uuid_file(self):
        return self.root / self.uuid_file_skel.format(self.version)

    @property
    def data(self):
        """
        data (easydict)

        data.config (list of python dicts)
        data.uuid (list of uuids)

        data.config[i] is a list of (key,values) corresponding to data.uuid[i]
        """
        return read_uuid_file(self.uuid_file)

    def init_uuid_file(self):
        if VERBOSE: print(f"Init [{cfg.uuid_file}]")
        if self.uuid_file.exists(): return None    
        data = edict({'uuid':[],'config':[]})
        write_uuid_file(self.uuid_file,data)

    def get_uuid(exp_config):
        uuid = get_uuid_from_config(self.data,exp_config)    
        if uuid == -1:
            if VERBOSE: print("Creating a new UUID and adding to cache file.")
            uuid = str(uuid_gen.uuid4())
            new_pair = edict({'uuid':uuid,'config':exp_config})
            append_new_pair(self.data,self.uuid_file,new_pair)
            return uuid
        else:
            if VERBOSE: print(f"Exp Config Already has a UUID {uuid}")
            return uuid
    
    def append_new(self,new_field,new_data):
        data = self.read_uuid_file()
        cont = set_new_field_default(data,new_field,default)
        if cont == -1:
            print(f"Not appending new field [{new_field}]. Field already exists.")
            return None
        set_new_field_data(data,new_data,new_field)
        self.version = self.version + 1
        write_uuid_file(self.uuid_file,data)
        print(f"Upgraded UUID cache version from v{self.version-1} to v{self.version}")
        


