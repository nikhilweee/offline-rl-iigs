# Offline RL for POMDPs

## Requirements
```console
$ pip install open_spiel
```

## Documentation
* `leduc_fp.py` runs fictitious play on leduc poker for 1000 iterations and collects trajectories using three different strategies:
    * `expert` collects 1M trajectories at the end of training, so this only has expert observations.
    * `mixed-lin` collects 1000 trajectories after every iteration, so this has mixed observations spread linearly.
    * `mixed-exp` collects varying number of samples after each iteration according to an exponential decay.
* `leduc_bc.py`  a simple MLP using observations collected from one of the strategies above.
* `submit_bc.sh` and `submit_fp.sh` are sbatch scripts to run the scripts on NYU Greene.