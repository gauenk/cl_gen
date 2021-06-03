
# -- python --
from pathlib import Path
from easydict import EasyDict as edict


# -- settings --
from settings import ROOT_PATH

class Config():

    def __init__(self,base,root_version,uuid_version,root_skel,uuid_file_skel):
        self._base = base
        self.root_version = root_version
        self.uuid_version = uuid_version
        self._root_skel = root_skel
        self._uuid_file_skel = uuid_file_skel

    @property
    def root(self):
        filename = self._base / Path(self._root_skel.format(self.root_version))
        return filename

    @property
    def uuid_file(self):
        filename = self.root / Path(self._uuid_file_skel.format(self.uuid_version))
        return filename

version = "1p3"
root_version = version
uuid_version = version
explore_root = Path(ROOT_PATH) / "./output/explore"
explore_package = "lpas"

base = explore_root / explore_package
root_skel = "{:s}"
uuid_file_skel = "uuid_database_{:s}.json"
cfg = Config(base,root_version,uuid_version,root_skel,uuid_file_skel)
