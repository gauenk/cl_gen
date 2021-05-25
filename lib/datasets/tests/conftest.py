import sys,os
from pathlib import Path
cwd = Path(os.path.abspath(__file__))
basename = "cl_gen"
base = None
for parent in cwd.parents:
    if parent.stem == basename:
        base = parent
        break
path = base / "lib"
sys.path.append( str(path) )
#os.chdir(cwd.parent)
