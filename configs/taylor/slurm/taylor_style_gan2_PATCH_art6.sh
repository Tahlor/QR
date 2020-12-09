#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem 10000
#SBATCH --ntasks 7
#SBATCH --output="taylor_style_gan2_PATCH_art6.slurm"
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

python -u train.py --config "./configs/taylor/cf_taylor_style_gan2_PATCH_art6.json" 

