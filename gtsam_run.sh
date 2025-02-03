#!/bin/bash
docker run -it \
    # --mount type=volume, src=gtsam-python-src, dst=/usr/src/external \
    --mount type=bind, dst=/usr/src, readonly \
    gtsam-python