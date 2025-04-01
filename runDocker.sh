#!/bin/bash
docker run -d --ipc=host --gpus all --user $(id -u):$(id -g) -it --rm  -v /home/username:/home/username -p 5000:8888 deepanepose-docker
