#!/bin/bash
PATH_DSET="PATH_TO_DSET"

xhost +local:root

docker run -it \
-e DISPLAY \
-e QT_x11_NO_MITSHM=1 \
--net=host \
-v ${PWD}:${PWD} \
-w ${PWD} \
-v $PATH_DSET:/data \
--gpus all \
-v /tmp/.X11-unix:/tmp/.X11-unix \
--name gpnet-data \
gpnet-data:latest

xhost -local:root
