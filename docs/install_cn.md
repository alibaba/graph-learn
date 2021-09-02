# 安装部署

## Wheel包安装

```bash
pip install graphlearn
```

## 源码编译安装

我们以```Ubuntu 16.04```下基于```g++ 5.4.0```为例, 来说明源码编译的步骤。


### 安装git

```bash
sudo apt-get install git-all
```

### 安装依赖的三方库

```bash
sudo apt-get install autoconf automake libtool libssl-dev cmake python-numpy python-setuptools python-pip
```

### 编译

首先，下载源代码:
```bash
git clone https://github.com/alibaba/graph-learn.git
cd graph-learn
git submodule update --init
```
接着，可以使用如下两种方式编译整个项目及测试用例：
1. 使用Makefile:
```bash
make test
```
2. 使用CMakeLists.txt:
```bash
mkdir cmake-build && cd cmake-build
cmake -DTESTING=ON .. && make
```
最后，编译python包，同时支持python2，python3:
```bash
make python
```
如需要执行特定的python bin，如指定用python3.7编译，则执行：
```bash
make python PYTHON=python3.7
```

### 安装

```bash
sudo pip install dist/your_wheel_name.whl
```

### (Optional) 安装TensorFlow
**GL**提供的Tensorflow模型示例基于**TensorFlow 1.12**开发，需要安装对应版本的库。
```bash
sudo pip install tensorflow==1.12.0
```

### (Optional) 安装PyTorch，PyG
**GL**提供的PyTorch模型示例基于**PyG**开发，需要安装对应的库。

```bash
sudo pip install pytorch
# Install PyG follow the doc: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
```

### 运行单元测试

```bash
source env.sh
./test_cpp_ut.sh
./test_python_ut.sh
```

## Docker镜像

[Graphlearn Docker hub](https://hub.docker.com/r/graphlearn/graphlearn)

我们提供了graphlearn的镜像，预装了对应版本的graphlearn。<br />
根据算法开发的需求，分别提供预装Tensorflow1.12.0和Pytorch1.8.1+PyG的镜像。<br />
您可以在Docker镜像中快速开始GraphLearn的运行。<br />

1. Tensorflow1.12.0, CPU

```bash
docker pull graphlearn/graphlearn:1.0.0-tensorflow1.12.0-cpu

# or, pull the latest graphlearn with tensorflow1.12.0-cpu
docker pull graphlearn/graphlearn:latest

# or, pull the given version graphlearn with tensorflow1.12.0-cpu
docker pull graphlearn/graphlearn:1.0.0

```

2. PyTorch1.8.1, Cuda10.2, cdnn7, with PyG

```bash
docker pull graphlearn/graphlearn:1.0.0-torch1.8.1-cuda10.2-cdnn7
```
