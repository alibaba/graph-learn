# 源码编译安装

我们以```Ubuntu 16.04```下基于```g++ 5.4.0```为例, 来说明源码编译的步骤。


## 安装git

```bash
sudo apt-get install git-all
```

## 安装依赖的三方库

```bash
sudo apt-get install autoconf automake libtool libssl-dev cmake python-numpy python-setuptools python-pip
```

## 编译

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
最后，编译python包:
```bash
make python
```

## 安装

```bash
sudo pip install dist/your_wheel_name.whl
```

## 安装TensorFlow
目前，**GL**提供的模型示例基于**TensorFlow 1.12**开发，需要安装对应的版本。
```bash
sudo pip install tensorflow==1.12.0
```

## 运行测试用例

```bash
source env.sh
./test_cpp_ut.sh
./test_python_ut.sh
```

