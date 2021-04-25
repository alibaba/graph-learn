# pip安装

我们在```Ubuntu 16.04```下基于```g++ 5.4.0```编译好了一个安装包，
如果和你的环境匹配，可以直接下载安装。否则参考“源码编译安装”。
目前，**GL**提供的模型示例基于**TensorFlow 1.12**开发，需要安装对应的版本。
只依赖系统接口做模型开发的用户，可以对源码稍作修改，去掉```__init__.py```文件中```import *tf*```相关的部分。

```bash
wget http://graph-learn-whl.oss-cn-zhangjiakou.aliyuncs.com/graphlearn-0.1-cp27-cp27mu-linux_x86_64.whl
sudo pip install graphlearn-0.1-cp27-cp27mu-linux_x86_64.whl
sudo pip install tensorflow==1.12.0
```

# 源码编译安装

## 安装git

```bash
sudo apt-get install git-all
```

## 安装依赖的三方库

```bash
sudo apt-get install autoconf automake libtool cmake python-numpy
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

## 运行测试用例

```bash
source env.sh
./test_cpp_ut.sh
./test_python_ut.sh
```

# Docker镜像

若使用Docker运行，可以下载我们准备好的镜像，也可以基于此镜像开发。

CPU版本
```
docker pull registry.cn-zhangjiakou.aliyuncs.com/pai-image/graph-learn:v0.1-cpu
```

GPU版本
```
docker pull registry.cn-zhangjiakou.aliyuncs.com/pai-image/graph-learn:v0.1-gpu
```
