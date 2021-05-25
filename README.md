# RLShield

This is RLShield, a prototype for experimenting with shielding based on almost-sure reachability in POMDPs.
It uses the model checker Storm.

RLShield has been used for the experiments in 

- [1] Enforcing Almost-sure Reachability in POMDPs by Sebastian Junges, Nils Jansen and Sanjit A. Seshia. CAV 2021.

RLShield integrates with [gridstorm](https://github.com/sjunges/gridworld-by-storm) to visualise gridworlds,
but can also be used standalone.

## Set-up
We provide a Docker image for easy testing, as well as some notes to install RLShield from source.

### Docker container
We provide a docker container
```
docker pull sjunges/rlshield:cav21
```

The container is based on an container for the probabilistic model checker as provided by the Storm developers, for details, see [this documentation](https://www.stormchecker.org/documentation/obtain-storm/docker.html).

The following command will run the docker container (for Windows platforms, please see the documentation from the storm website).
```
docker run --mount type=bind,source="$(pwd)",target=/data -w /opt/premise --rm -it --name premise sjunges/premise:cav21
```

Files that one copies into /data are available on the host system in the current working directory.

You will see a prompt inside the docker container.

### Installation/Dependencies

Users of an artefact/Docker container can skip this step.

- Install Storm and Stormpy from source [as usual](https://moves-rwth.github.io/stormpy/installation.html).
    - but using [this repository for Storm](https://github.com/sjunges/storm/tree/prismlang-sim)
    - and [this repository for Stormpy](https://github.com/sjunges/stormpy/tree/prismlang-sim)
    (Note: These branches are in the process of being merged back into Storm(py) version 1.7.0)
- Install ffmpeg (with your package manager)
- run `python setup.py install`

## Running RLShield on Gridstorm examples

The basic usage of the tool can be reproduced as follows: 
```
python rlshield/shield.py -m refuel --constants "N=6,ENERGY=8" --video-path .  
```
The model states that the grid-model `refuel` should be used with constants as specified. 
We run one episode and store videos in the current working directory.

The output will be `refuel-6-8-computed-shield-0.mp4`. A logfile is written to `rendering.log`

Rather than running a single episode, we can run multiple episodes:
```
python rlshield/shield.py -m refuel --constants "N=6,ENERGY=8" -N 5 --video-path .  
```

We observe that episodes have different lengths before they are 'finished', some episodes are never finished.
By default, episodes are restricted to 100 steps. The number of episodes requested is the number of episodes that finish.
It is helpful to limit the number of episodes, even if that means that we would not get 5 videos:
```
python rlshield/shield.py -m refuel --constants "N=6,ENERGY=8" -N 5 --max-runs 6 --maxsteps 75 --video-path .  
```

Some further commands
- For reproducibility, RLShield currently fixes the seed. You can override the seed with `--seed X`
- To run different models from gridstorm, see the possible arguments for `-m`, see also [1, Table 1] for meaningful experiments.
- To not compute a shield, add `--noshield`
- The logfile can be set with `--logfile PATH`

### Precomputed shields

Storm can precompute shields with the `--export-winningregion` option. 
These winning regions can be loaded: 
```
python rlshield/shield.py -m refuel --constants "N=6,ENERGY=8"  --video-path .  --load-winning-region examples/refuel-6-8.wr
```
We notice that no proper checks are done to ensure that the winning region matches the model. 

### Collect statistics

Rather than visualising traces, we can also collect some statistics. 
```
python rlshield/shield.py -m refuel --constants "N=6,ENERGY=8" -N 5 --max-runs 6 --max-steps 75 --stats-path .  
```
This will write statistics to the current working directory. 

We can currently not do both statistics and videos in one run, but as the seed is fixed, the statistics and videos will match. 

## Running RLShield on own examples

RLShield can be run on other benchmarks. We give a trivial toy example.
We notice that this support is more experimental. 

We consider the following maze, in which 11 is the goal, and 12 and 14 are to be avoided. (This is a variant of the cheesemaze).
Storm gives the following internal IDs to states:
```
 1  2  3  4  5
 6     7     8
 9     10   11
12     13   14
```
Running 
```
python rlshield/shield.py --prism examples/maze2.prism --prop "Pmax=? [!\"bad\" U \"goal\"]" --constants "sl=0.3" -N 5 --noshield
```
will show a series of traces. Some of these traces do indeed visit states 12 or 14.
 
Running 
```
python rlshield/shield.py --prism examples/maze2.prism --prop "Pmax=? [!\"bad\" U \"goal\"]" --constants "sl=0.3" -N 5
```
shows that these states are no longer visited. 