# PIC & FLIP fluid simulation
This is the code of the final project for the course Physically-Based Simulation in Computer Graphics, ETHZ, HS2021.
We use [Taichi](https://github.com/taichi-dev/taichi) to implement a fluid solver as described in the paper: *Zhu, Yongning and Bridson, Robert, Animating sand as a fluid. ACM
Transactions on Graphics 2005, 24, pp. 965-972*.

## Cloning the repository

```bash
git clone --recursive https://github.com/boelukas/pbs-sandfluid.git
git submodule update --init --recursive
```

## Setting up the environment

```bash
conda env create -f env.yaml
conda activate sandfluid
```

## Running the code

```bash
python src/simulation.py
```

## Controlling the GUI

|Key|Action|
|-|-|
|*space*| start/pause simulation|
|*r*| reset simulation|


## Results

https://user-images.githubusercontent.com/48366676/145845275-47e32fd9-0171-44dc-9185-a0f8578e5e36.mp4


https://user-images.githubusercontent.com/48366676/145851018-ceed7f97-ee2b-445a-beef-85d8119073e1.mp4

