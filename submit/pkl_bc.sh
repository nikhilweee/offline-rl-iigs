#!/bin/bash
# Add -e to stop on errors
# Add -x to print before executing

#SBATCH --mem=32GB
#SBATCH --time=02-00:00:00
#SBATCH --output="logs/%A_%a_%x.txt"
#SBATCH --job-name=bc
#SBATCH --array=0,1,2,3

# The following options will not be applied
# SBATCH --cpus-per-task=1
# SBATCH --nodelist="gr*"
# SBATCH --gres=gpu:rtx8000:1
# SBATCH --gres=gpu:1

trajs=("936-4609-40157" "936-8838-40520" "936-9249-79840" "936-9457-82042")
traj=${trajs[${SLURM_ARRAY_TASK_ID}]}

singularity exec \
    --overlay /scratch/nv2099/images/overlay-50G-10M.ext3:ro \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh;
    conda activate mcs;
    cd /home/nv2099/projects/mcs/pkl_obs;
    python -u leduc_bc_offline.py --traj trajectories/traj-mixed-${traj}.pkl --label ${traj}/ep-100k-lr-1e5;
    "
