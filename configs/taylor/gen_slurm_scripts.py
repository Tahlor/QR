import os
import json
from pathlib import Path
from easydict import EasyDict as edict
import warnings

def get_root():
    file_dir = Path(os.path.dirname(__file__)).resolve()
    for par in file_dir.parents:
        if par.parent.name =="fslg_qr":
            return par
    warnings.warn("Couldn't find parent")
    return "."

ROOT = get_root()
ENV = "/zgrouphome/fslg_qr/env/qr"
Path(f"./auto/scripts").mkdir(parents=True, exist_ok=True)

def create_script(config_path, outpath="./slurm"):
    config_name = config_path.stem.replace("___","").replace("cf_","")
    checkpoint = ROOT / Path(f"saved/{config_name}/checkpoint-latest.pth")
    if checkpoint.exists():
        checkpoint = f"./saved/{config_name}/checkpoint-latest.pth"
        resume = f"""--resume "{checkpoint}" """
    else:
        resume = ""
    print(checkpoint, "resume command", resume)
    script = f"""#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem 10000
#SBATCH --ntasks 7
#SBATCH --output="{config_name}.slurm"
#SBATCH --time 36:00:00
#SBATCH --mail-user=taylornarchibald@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

# module purge
# module load cuda/10.1
# module load cudnn/7.6
module load gcc/7
module load cuda/9.2
module load cudnn/7.1

export PATH="/zgrouphome/fslg_qr/env/qr:$PATH"
eval "$(conda shell.bash hook)"
conda activate "/zgrouphome/fslg_qr/env/qr"

cd "/lustre/scratch/grp/fslg_qr/qr2"
which python

python -u train.py --config "./configs/taylor/{config_path.as_posix()}" {resume}

"""
    out_path = Path(f"{outpath}/{config_name}.sh")
    out_path.open("w").write(script)
    os.chmod(out_path, 0o755)

def check_config(path):
    correct_name = path.stem.replace("___", "")
    print(correct_name)
    config = edict(json.load(path.open()))
    if config.name != correct_name:
        config.name = correct_name
        locals().update(globals())
        json.dump(config.__dict__, path.open("w"), indent=4, separators=(',', ':'))

def run_it():
    for f in Path(".").glob("*.json"):
        check_config(f)
        create_script(f)

if __name__=='__main__':
    run_it()
