#!/bin/bash
docker run -d --ipc=host --gpus all --user $(id -u):$(id -g) -it --rm  -v /home/kerrien:/home/kerrien/data -p 5000:8888 deepanepose-docker
