# 安装部署

## Wheel包安装

```bash
pip install graph-learn
```

## 源码编译安装

我们以``Ubuntu 20.04`` based on ``g++ 9.4.0``为例, 来说明源码编译的步骤。


### 安装依赖库

```bash
sudo apt-get -y update
python -m pip install --upgrade pip setuptools wheel
pip install numpy

git clone https://github.com/alibaba/graph-learn.git
cd graph-learn
cd graphlearn
sudo ./install_dependencies.sh
```

### 编译

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

### 安装python wheel包

```bash
pip install dist/your_wheel_name.whl
```

### (可选) 安装TensorFlow
GL提供的Tensorflow模型示例基于TensorFlow 1.13开发，需要安装对应版本的库。
```bash
sudo pip install tensorflow==1.13.0
```

### (可选) Install PyTorch，PyG
GL提供的PyTorch模型示例基于PyG开发，需要安装对应的库。
```bash
sudo pip install pytorch
# Install PyG follow the doc: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
```

## Docker镜像

[Graphlearn Docker hub](https://hub.docker.com/r/graphlearn/graphlearn)

我们提供了一个ubuntu20.04+gcc9.4.0的docker镜像

```bash
docker pull graphlearn/graph-learn:1.1.0
```
