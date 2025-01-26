FROM ubuntu:22.04

RUN apt-get -y update && apt-get -y install
RUN apt-get -y install libboost-all-dev cmake
RUN apt-get -y install libtbb-dev
RUN apt-get -y update && apt-get -y install

WORKDIR /usr/src/

RUN git clone --single-branch --branch develop https://github.com/borglab/gtsam.git

WORKDIR /usr/src/gtsam/build


RUN cmake -DCMAKE_BUILD_TYPE=Release \
    -DGTSAM_WITH_EIGEN_MKL=OFF \
    -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_BUILD_TIMING_ALWAYS=OFF \
    -DGTSAM_BUILD_TESTS=OFF \
    -DGTSAM_BUILD_PYTHON=1 \
    -DGTSAM_PYTHON_VERSION=3.10.12 \
    ..
