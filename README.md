# radarmeetsvision
This repository contains the code of the ICRA 2025 submission (in review) "Radar Meets Vision: Robustifying Monocular Metric Depth Prediction for Mobile Robotics". The submitted version can be found on [arxiv](https://arxiv.org/abs/2410.00736).

## Setup
The easiest way to get started is using the respective devcontainer.
The devcontainer all dependencies installed, including a patch that allows rendering depth images without a screen attached to the host.
To use the devcontainer, install Visual Studio Code and follow the [instructions](https://code.visualstudio.com/docs/devcontainers/containers).
In short, only vscode and docker are required.
If installing directly on the host machine is more desirable, the Dockerfiles are a good starting point for the required dependencies.

## radarmeetsvision
This directory contains the network in a python package format. It can be installed using `pip install .` .

### Interface
The file `interface.py` aims to provide an interface to the network.
In this file, most methods to train and evaluate are provided.
In the `scripts` directory, usage examples are provided.
To reproduce the training and evaluation results, the two scripts `train.py` and `evaluate.py` can be run.
For details on the process, we refer to the paper, but in general, no adjustements are needed.

### metric_depth_network
Contains the metric network based on [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).
All important code diffs compared to this upstream codebase are outlined in general.

### Pretrained networks
We provide the networks obtained from training and used for evaluation in the paper [here](https://drive.google.com/file/d/1jJN_75OLDWyFMOjH_NrAl0Set08Gpzh0/view?usp=drive_link).

## Training datasets
**blearn** is a tool that allows you to generate a synthetic image and depth training dataset using Blender.
Together with a mesh and texture obtained from photogrammetry, realistic synthetic datasets can be generated.
The script is executed with Blender's built-in Python interpreter, which has the advantage that the Blender Python API is correctly loaded already.

### Download existing datasets
Existing datasets can be downloaded from [here](https://drive.google.com/file/d/1jJN_75OLDWyFMOjH_NrAl0Set08Gpzh0/view?usp=drive_link). The download contains the datasets, as well as the blender projects used to obtain the datasets.

### Generating training datasets
In order to (re-)generate the training datasets, the following steps are needed:
1. Obtaining .fbx mesh and texture for the area of interest. We use aerial photogrammetry and Pix4D to create this output.
2. Import the fbx file into a new blender project. Ensure that only the following elements are present in the blender project: A mesh, a camera and a sun light source. As convenience, we provide the blender projects used in the paper.
3. Create a configuration file: Easiest is to start from an existing configuration file. The main values to adjust are the camera intrinsics, as well as the extent of the mesh, i.e. `paths`: `xmin`, `xmax`, ... which defines the area which is sampled on the mesh and the number of samples the dataset should contain. One thing to consider is the position and orientation with respect to the blender origin. The z-coordinate specification in this tool is always with respect to the distance to ground, essentially doing terrain following. If you are using one of the provided meshes, then the extent does not need to be adjusted.
4. The dataset rendering can then be started using: `blender -b <path to blender project file> --python blearn.py`. Ensure that you adjust the config file accordingly in `blearn.py`.

## Validation datasets
The method for obtaining the validation datasets is described in the paper. The datasets are made available [here](https://drive.google.com/file/d/1jJN_75OLDWyFMOjH_NrAl0Set08Gpzh0/view?usp=drive_link).