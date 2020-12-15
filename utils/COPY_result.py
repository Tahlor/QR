import json
import shutil
from pathlib import Path
from easydict import EasyDict as edict
import socket
import os
from distutils.dir_util import copy_tree
copy = copy_tree #shutil.copy

def get_root():
    file_dir = Path(os.path.dirname(__file__)).resolve()
    for par in file_dir.parents:
        if par.parent.name =="fslg_qr":
            return par
        elif par.stem.lower()[:2]=="qr":
            return par
    #warnings.warn("Couldn't find parent")
    print(file_dir)
    raise Exception("Couldn't find parent")

    return "."

os.chdir(get_root())

def get_computer():
    return socket.gethostname()
def is_galois():
    return get_computer() == "Galois"

if is_galois():
    dataset_root = Path("/media/taylor/Seagate Backup Plus Drive/datasets")
else:
    dataset_root = Path("../data")

ROOT = get_root()

SOURCE = Path("/media/data/GitHub/qr2")
JSON_PATH = SOURCE / "saved/GOOD_just_paths/___GOOD_just_paths.json"

DESTINATION = Path("/home/taylor/shares/SuperComputerRoot/lustre/scratch/grp/fslg_qr/qr2")
# OUTPUT_CONFIG = DESTINATION / "configs" / "copied"
# OUTPUT_CONFIG.mkdir(exist_ok=True,parents=True)
config = edict(json.load(Path(JSON_PATH).open()))


config.data_loader.data_dir = str(dataset_root / Path(config.data_loader.data_dir).name)
name = config.name

## PRINT DIR
d = str(DESTINATION / f"train_out/{name}")
copy(config.trainer.print_dir, d)
config.trainer.print_dir = d

## CACHE DIR
d = str(DESTINATION / f"cache/{name}")
copy(config.sample_data_loader.cache_dir, d)
config.sample_data_loader.cache_dir = d

## SAVE DIR
copy(str(SOURCE / "saved" / name), str(DESTINATION / "saved" / name))

json.dump(config.__dict__, (DESTINATION / "saved" / f"___{name}.json").open("w"), indent=4, separators=(',', ':'))

# Path(config.trainer.print_dir).mkdir(parents=True, exist_ok=True)
# Path(config.sample_data_loader.cache_dir).mkdir(parents=True, exist_ok=True)
