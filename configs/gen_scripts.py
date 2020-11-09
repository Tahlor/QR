from pathlib import Path
ENV = "/zgrouphome/fslg_qr/env/qr"
Path(f"./auto/scripts").mkdir(parents=True, exist_ok=True)
def create_script(config_path):
    script = f"""#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem 10000
#SBATCH --ntasks 7
#SBATCH --output="{config_path.stem}.slurm"
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

cd "/fslhome/tarch/fsl_groups/fslg_qr/qr"
which python

python -u train.py --config "./configs/{config_path.as_posix()}"

"""
    Path(f"./auto/scripts/{config_path.stem}.sh").open("w").write(script)

for f in Path("./auto").glob("*.json"):
    create_script(f)