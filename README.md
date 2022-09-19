# Dataset generation for GP-net

This repository includes code for generating data for GP-net. If you want to use GP-net with a PAL parallel jaw
gripper, you can download the pre-trained model or the training data from 
[zenodo](https://zenodo.org/record/7092009#.YyghmtXMJl8) and will not need to use this code.

If you want to train GP-net for an alternative gripper, you can generate data with this codebase and use it
to train a model with [GP-net](https://github.com/AuCoRoboticsMU/GP-net).

## Installation

We use the [DexNet](https://github.com/BerkeleyAutomation/dex-net) python package to generate our data. Several
modifications of the original code have been made to apply it to the use-case of mobile robots and 6-DOF grasps.

We highly recommend the installation via [docker](https://www.docker.com). A pre-built docker image is 
available on [zenodo](https://zenodo.org/record/7092009#.YyghmtXMJl8). After you downloaded and unpacked it on your
machine, you can use `./run_docker.sh` to run the docker image. Note that you have to change PATH_DSET 
in `run_docker.sh` to the directory path where you will store your meshes and dataset. It will be mounted to `/data` 
in the docker container.

## Generating a dataset for GP-net

1. Download the object meshes from 3dnet and kit from the DexNet platform 
on [box.com](https://berkeley.app.box.com/s/w6bmvvkp399xtjpgskwq1cytkndmm7cn) and store them 
in a folder called `raw_meshes`.
2. Change the name of the dataset, the gripper name and the path to `raw_meshes` in the `cfg/tools/config.yaml` file.
Note that the gripper name has to be an available gripper in the `data/grippers/` directory. You can add your own gripper
configuration if needed.
2. Sample grasps for the hdf5 database by running `python tools/sample_grasps.py`. This code will loop through the kit and 3d meshes, sample grasps and
   store them in a hdf5 database.
3. Render images and store the grasp information in a dataset
   1. Change the environment parameters in `cfg/tools/render_dataset.yaml` if needed
   3. Run `python tools/render_dataset.py` to generate your dataset
4. Generate the indices of an object-wise split by running `python tools/split_indices_object_wise.py`
5. Add kinetic noise to the depth images by running `python tools/apply_noise_to_depth_images.py`
6. Done! The dataset can now be used to train a GP-net model.


------

If you use this code, please cite

A. Konrad, J. McDonald and R. Villing, "Proposing Grasps for Mobile Manipulators," in review.

J. Mahler, J. Liang, S. Niyaz, M. Laskey, R. Doan, X. Liu, J. A. Ojea,
and K. Goldberg, “Dex-net 2.0: Deep learning to plan robust grasps with
synthetic point clouds and analytic grasp metrics,” in Robotics: Science
and Systems (RSS), 2017.

A. Handa, T. Whelan, J. McDonald and A. Davison, "A benchmark for RGB-D visual odometry, 
3D reconstruction and SLAM," in Internation Conference on Robotics and Automation (ICRA), 2014.