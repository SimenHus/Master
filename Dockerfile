################################################################
# Build basic gtsam image
# Inspired and copied from https://github.com/cntaylor/gtsam-python-docker-vscode
FROM ubuntu:22.04 AS gtsam-basic-libraries

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y install
RUN apt-get -y install build-essential cmake
RUN apt-get -y install libboost-all-dev
RUN apt-get -y install libtbb-dev
RUN apt-get -y install git
RUN apt-get -y install python3-pip python3-dev
RUN apt-get -y update && apt-get -y install

# Get python packages
RUN python3 -m pip install matplotlib numpy pyparsing


################################################################
# Image including gtsam

FROM gtsam-basic-libraries AS gtsam-downloaded

WORKDIR /usr/src/

RUN git clone --single-branch --branch develop https://github.com/borglab/gtsam.git

################################################################
#  Building docs
FROM gtsam-downloaded AS gtsam-docs

RUN apt-get -y install doxygen graphviz

WORKDIR /usr/src/gtsam/build

RUN cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DGTSAM_WITH_EIGEN_MKL=OFF \
    -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_BUILD_TIMING_ALWAYS=OFF \
    -DGTSAM_BUILD_TESTS=OFF \
    -DGTSAM_BUILD_PYTHON=ON \
    -DGTSAM_PYTHON_VERSION=3 \
    .. \
    && make doc

RUN apt-get install -y apache2
# Put docs where apache can locate it
RUN rm -r /var/www/html && cp -r /usr/src/gtsam/doc/html /var/www

# Start webserver
EXPOSE 80
CMD ["apache2ctl", "-D", "FOREGROUND"]


################################################################
# Build gtsam end state
FROM gtsam-downloaded AS gtsam-build

# Change to build directory. 
WORKDIR /usr/src/gtsam/build
# Run cmake
RUN cmake \
    # Can switch the following to Debug if you need to "step into" GTSAM libraries
    -DCMAKE_BUILD_TYPE=Release \
    -DGTSAM_WITH_EIGEN_MKL=OFF \
    -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_BUILD_TIMING_ALWAYS=OFF \
    -DGTSAM_BUILD_TESTS=OFF \
    -DGTSAM_BUILD_PYTHON=ON \
    -DGTSAM_PYTHON_VERSION=3\
    ..

# Build and install gtsam code
RUN make -j4 python-install
RUN make -j4 install
RUN make clean

# Needed to link with GTSAM
ENV LD_LIBRARY_PATH=/usr/local/lib

CMD ["bash"]