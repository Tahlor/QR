import os
import json
from pathlib import Path
from easydict import EasyDict as edict

ENV = "/zgrouphome/fslg_qr/env/qr"
Path(f"./auto/scripts").mkdir(parents=True, exist_ok=True)
def create_script(config_path, outpath="./slurm"):
    config_name = config_path.stem.replace("___","")
    checkpoint = Path(f"../saved/{config_name}/checkpoint-latest.pth")
    if checkpoint.exists():
        checkpoint = f"./saved/{config_name}/checkpoint-latest.pth"
        resume = f"""--resume "{checkpoint}" """
    else:
        resume = ""
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

module purge
module load cuda/10.1
module load cudnn/7.6

export PATH="{ENV}:$PATH"
eval "$(conda shell.bash hook)"
conda activate "{ENV}"

cd "/lustre/scratch/grp/fslg_qr/qr"
which python

python -u train.py --config "./configs/{config_path.as_posix()}" {resume}

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