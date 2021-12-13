# Install Instructions

## Prerequisites
To run the simulation the packages in env.yml need to be installed.

Additionally [Taichi](https://github.com/taichi-dev/taichi) needs to be downloaded and locally installed.
This can be done by cloning the Github [repository](https://github.com/boelukas/pbs-sandfluid), which has Taichi added as submodule.

## Creating the environment

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
