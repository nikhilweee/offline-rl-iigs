#!/bin/bash

#SBATCH --mem=32GB
#SBATCH --time=00-06:00:00
#SBATCH --output="logs/%A_%a_%x.txt"
#SBATCH --job-name=env_mlp
#SBATCH --gres=gpu:1

echo "Starting SLURM Script"

#SBATCH --gres=gpu:rtx8000:1
#SBATCH --nodelist="*gr*"
#SBATCH --array=0,1,2,3
#SBATCH --cpus-per-task=1

# nums=("100" "50" "25" "10")
# num=${nums[${SLURM_ARRAY_TASK_ID}]}

singularity exec --nv \
    --overlay /scratch/nv2099/images/overlay-50G-10M.ext3:ro \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh;
    conda activate spiel;
    cd /home/nv2099/projects/mcs/pkl_obs;
    python -u env_mlp.py --label mlp
    "
