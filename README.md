# Offline RL for POMDPs

## Requirements
```console
$ pip install open_spiel
```

## Description
* `leduc_fp.py` runs fictitious play on leduc poker for 1000 iterations and collects trajectories using three different strategies:
    * `expert` collects 1M trajectories at the end of training, so this only has expert observations.
    * `mixed-const` collects 1000 trajectories after every iteration, so this has mixed observations sampled at a constant rate.
    * `mixed-exp` collects varying number of samples after each iteration according to an exponential decay.
* `leduc_bc.py` is a simple MLP using observations collected from one of the strategies above.
* `leduc_crr.py` runs tabular CRR using observations collected from one of the strategies above.
* `observation.py` holds the observation data structure for easy conversion to / from CSVs.
* `submit_bc.sh` / `submit_fp.sh` are sbatch scripts to run the scripts on NYU Greene.

## Workthrough
* First, collect observations using one of the three strategies
```console
$ python leduc_fp.py --mode expert --num_iterations 100 --num_episodes 1_000 --traj trajectories-1k.csv
```
* Next, you can run behaviour cloning using the following command
```console
$ python leduc_bc.py --traj trajectories-1k.csv
```
* You can also run tabular CRR using the following command. Note that this is still very much work in progress.
```console
$ python leduc_crr.py --traj trajectories-1k.csv
```
