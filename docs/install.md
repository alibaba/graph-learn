# Install from pip

We build the wheel package with```g++ 5.4.0``` on ```Ubuntu 16.04```.
Just download and install it if it matches your environment.
Otherwise, please refer to the section 'build from source'.

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
sudo apt-get install autoconf automake libtool cmake
pip install numpy
```

## Compile
```bash
git clone https://github.com/alibaba/graph-learn.git
cd graph-learn
git submodule update --init
make test
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

You can download our docker image to execute **GL** projects.

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
