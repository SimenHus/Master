
https://github.com/borglab/gtsam/blob/develop/python/README.md
https://github.com/borglab/wrap


Installation of dependancies:

Create a python environment
python -m venv ~/py_envs/gtsam-3.10

Source python venv
source ~/py_envs/gtsam-3.10/bin/activate

Install python dependencies:
pip install matplotlib numpy pyparsing
pip install pybind11 pybind11-stubgen

Download gtsam in desired folder
git clone --single-branch --branch develop https://github.com/borglab/gtsam

Make build folder in downloaded repo, go into build folder
mkdir build
cd build

Set up cmake variables

cmake -DCMAKE_BUILD_TYPE=Release -DGTSAM_WITH_EIGEN_MKL=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_TIMING_ALWAYS=OFF -DGTSAM_BUILD_TESTS=OFF -DGTSAM_UNSTABLE_BUILD=OFF -DGTSAM_UNSTABLE_BUILD_PYTHON=OFF -DGTSAM_GENERATE_DOC_XML=ON -DGTWRAP_ADD_DOCSTRINGS=ON -DGTSAM_BUILD_PYTHON=ON -DGTSAM_PYTHON_VERSION=3 ..

Make python:
sudo make python-install

Make normal:
sudo make install

Clean:
sudo make clean




From gtsam repo:
The wrap library provides for building the Python wrapper with docstrings included, sourced from the C++ Doxygen comments. To build the Python wrapper with docstrings, follow these instructions:

Build GTSAM with the flag -DGTSAM_GENERATE_DOC_XML=1. This will compile the doc/Doxyfile.in into a Doxyfile with GENERATE_XML set to ON.
From the project root directory, run doxygen build/<build_name>/doc/Doxyfile. This will generate the Doxygen XML documentation in xml/.
Build the Python wrapper with the CMake option GTWRAP_ADD_DOCSTRINGS enabled.