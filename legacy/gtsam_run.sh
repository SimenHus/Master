#!/bin/bash
docker run -it \
    --mount type=bind,src=.,dst=/usr/src/external,readonly \
    gtsam-python