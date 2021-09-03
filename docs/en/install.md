# Install and Setup

## Install from wheel.

```bash
pip install graph-learn
```

## Build from source.

Let's take ``Ubuntu 16.04`` based on ``g++ 5.4.0`` as an example to illustrate the steps of compiling the source code.


### Install dependent libraries

```bash
sudo apt-get install git-all autoconf automake libtool libssl-dev cmake python-numpy python-setuptools python-pip
```

### Build

```bash
git clone https://github.com/alibaba/graph-learn.git
cd graph-learn
git submodule update --init
```
Next, the entire project and test cases can be compiled in two ways.
1. use Makefile(recommended):
```bash
make test
```
2. Use CMakeLists.txt:
```bash
mkdir cmake-build && cd cmake-build
cmake -DTESTING=ON .. && make
```

build python package:
```bash
make python
```
If you need to execute a specific python bin, such as specifying compilation with python 3.7, then execute:
```bash
make python PYTHON=python3.7
```

### Install

```bash
sudo pip install dist/your_wheel_name.whl
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

### Run UT.

```bash
source env.sh
./test_cpp_ut.sh
./test_python_ut.sh
```

## Docker images

[Graphlearn Docker hub](https://hub.docker.com/r/graphlearn/graphlearn)

We provide a graphlearn image with the corresponding version of graphlearn pre-installed.
Depending on the needs of algorithm development, we provide pre-installed images of Tensorflow 1.13.0rc1 and Pytorch 1.8.1+PyG, respectively.
You can quickly start GraphLearn in the Docker image.

1. Tensorflow1.13.0rc1, CPU

```bash
docker pull graphlearn/graphlearn:1.0.0-tensorflow1.13.0rc1-cpu

# or, pull the latest graphlearn with 1.0.0-tensorflow1.13.0rc1-cpu
docker pull graphlearn/graphlearn:latest

# or, pull the given version graphlearn with 1.0.0-tensorflow1.13.0rc1-cpu
docker pull graphlearn/graphlearn:1.0.0

```

2. PyTorch1.8.1, Cuda10.2, cdnn7, with PyG

```bash
docker pull graphlearn/graphlearn:1.0.0-torch1.8.1-cuda10.2-cudnn7
```
