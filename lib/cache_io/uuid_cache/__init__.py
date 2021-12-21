"""
UUID Cache

Converts a collection of python dictionaries
with many keys into a uuid string
to be saved into a file.

UUID_DATABASE =

   key_1    |   key_1    | ... |    uuid
  v_{1,1}   |   v_{1,2}  | ... |   uuid_{1}
  v_{2,1}   |   v_{2,2}  | ... |   uuid_{2}
  ...
  v_{N,1}   |   v_{N,2}  | ... |   uuid_{N}


--- Functionality ---

- dictionary <-> uuid
  - write uuid specific filename
  - read u
- compare two dictionaries

root_directory/
    uuid_database_{version}.json
    uuid_str_1
    uuid_str_2
    ...
    uuid_str_N

uuid_database_{version}.json stores the UUID_DATABASE (pic above)

"""


import json,pprint
import uuid as uuid_gen
from easydict import EasyDict as edict

from ._write import *
from ._read import *
from ._utils import *
from ._convert import *
from ._append import *
from ._debug import *

class UUIDCache():

    def __init__(self,root,version):
        self.root = root
        self.version = version
        self.uuid_file_skel = "uuid_database_{:s}.json"
        # self.init_uuid_file()

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

    def get_uuid_from_config(self,exp_config):
        if self.data is None:
            self.init_uuid_file()
            return -1
        else:
            return get_uuid_from_config(self.data,exp_config)

    def get_config_from_uuid(self,uuid):
        if self.data is None:
            self.init_uuid_file()
            return None
        else:
            return get_config_from_uuid(self.data,uuid)

    def get_config_from_uuid_list(self,uuids):
        configs = []
        for uuid in uuids:
            configs.append(self.get_config_from_uuid(uuid))
        return configs

    def init_uuid_file(self):
        if VERBOSE: print(f"Init [{uuid_file}]")
        if self.uuid_file.exists(): return None
        data = edict({'uuid':[],'config':[]})
        write_uuid_file(self.uuid_file,data)

    def get_uuid(self,exp_config):
        uuid = self.get_uuid_from_config(exp_config)
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
        data = self.data
        cont = set_new_field_default(data,new_field,default)
        if cont == -1:
            print(f"Not appending new field [{new_field}]. Field already exists.")
            return None
        set_new_field_data(data,new_data,new_field)
        self.version = self.version + 1
        write_uuid_file(self.uuid_file,data)
        print(f"Upgraded UUID cache version from v{self.version-1} to v{self.version}")

    def __str__(self):
        return f"UUIDCache version [{self.version}] with file at [{self.uuid_file}]"


