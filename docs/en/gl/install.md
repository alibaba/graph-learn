# Install and Setup

## Install from wheel.

```bash
pip install graph-learn
```

## Build from source.

Let's take ``Ubuntu 20.04`` based on ``g++ 9.4.0`` as an example to illustrate the steps of compiling the source code.


### Install dependent libraries

```bash
sudo apt-get -y update
python -m pip install --upgrade pip setuptools wheel
pip install numpy

git clone https://github.com/alibaba/graph-learn.git
cd graph-learn
cd graphlearn
sudo ./install_dependencies.sh
```

### Build

```bash
mkdir build
cd build
cmake -DGL_PYTHON_BIN=python ..
make -j
# cpp test
source env.sh
./test_cpp_ut.sh
```

build python package:
```bash
make python -j
# python test
pip install dist/*.whl
./test_python_ut.sh
```

### Install

```bash
pip install dist/your_wheel_name.whl
```

### (Optional) Install TensorFlow
The TensorFlow model example provided by **GL** is developed based on **TensorFlow 1.13** and requires the installation of the corresponding version of the library.
```bash
sudo pip install tensorflow==1.13.0
```

### (Optional) Install PyTorch，PyG
The PyTorch model example provided by **GL** is based on **PyG** development and requires the installation of the corresponding library.

```bash
sudo pip install pytorch
# Install PyG follow the doc: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
```

## Docker images

[Graphlearn Docker hub](https://hub.docker.com/r/graphlearn/graphlearn)

We provide a ubuntu20.04+gcc9.4.0 image.

```bash
docker pull graphlearn/graph-learn:1.1.0
```