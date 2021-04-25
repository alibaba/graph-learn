# Install from pip

We have built a wheel package with```g++ 5.4.0``` and ```python 2.7``` on ```Ubuntu 16.04```.
If it matches your environment, just download and install it directly.
Otherwise, please refer to the section 'Build from source'.
Currently, the examples provided by **GL** are developed based on **TensorFlow 1.12**. To run the examples, please install **TensorFlow 1.12** first.
Users that develop thier own models based on system interfaces can modify the source code slightly and remove the relevant part of ```import *tf*``` in the ```__init__.py``` file.

## Get wheel package

```bash
wget http://graph-learn-whl.oss-cn-zhangjiakou.aliyuncs.com/graphlearn-0.1-cp27-cp27mu-linux_x86_64.whl
```

## Install using pip

```bash
sudo pip install graphlearn-0.1-cp27-cp27mu-linux_x86_64.whl
```

## Install TensorFlow

```bash
sudo pip install tensorflow==1.12.0
```

# Build from source

## Install git

```bash
sudo apt-get install git-all
```

## Install dependent libraries

```bash
sudo apt-get install autoconf automake libtool libssl-dev cmake python-numpy python-setuptools python-pip
```

## Compile
First, get the source code:
```bash
git clone https://github.com/alibaba/graph-learn.git
cd graph-learn
git submodule update --init
```
To build the project with tests, you can use one of the following options:
1. Using Makefile:
```bash
make test
```
2. Using CMakeLists.txt:
```bash
mkdir cmake-build && cd cmake-build
cmake -DTESTING=ON .. && make
```
To build python package:
```bash
make python
```

## Install
```bash
sudo pip install dist/your_wheel_name.whl
```

## Run test
```bash
source env.sh
./test_cpp_ut.sh
./test_python_ut.sh
```

# Docker image

You can download our docker image to run **GL** projects.

CPU version:
```
docker pull registry.cn-zhangjiakou.aliyuncs.com/pai-image/graph-learn:v0.1-cpu
```
GPU version:
```
docker pull registry.cn-zhangjiakou.aliyuncs.com/pai-image/graph-learn:v0.1-gpu
```

You can also refer to our Dockerfile to build your own image. Please checkout this [document](../docker_image/README.md).

[Home](../README.md)
