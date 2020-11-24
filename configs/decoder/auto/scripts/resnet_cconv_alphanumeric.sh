#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem 10000
#SBATCH --ntasks 7
#SBATCH --output="resnet_cconv_alphanumeric.slurm"
#SBATCH --time 36:00:00
#SBATCH --mail-user=taylornarchibald@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge
module load cuda/10.1
module load cudnn/7.6

export PATH="/zgrouphome/fslg_qr/env/qr:$PATH"
eval "$(conda shell.bash hook)"
conda activate "/zgrouphome/fslg_qr/env/qr"

cd "/lustre/scratch/grp/fslg_qr/qr"
which python

python -u train.py --config "./configs/auto/___resnet_cconv_alphanumeric.json" --resume "./saved/resnet_cconv_alphanumeric/checkpoint-latest.pth" 

