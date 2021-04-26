# Build from source

We carried out experiments under ```g++ 5.4.0``` and ```python 2.7``` on ```Ubuntu 16.04```.

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

## Install TensorFlow

Currently, the examples provided by **GL** are developed based on **TensorFlow 1.12**. To run the examples, please install **TensorFlow 1.12** first.

```bash
sudo pip install tensorflow==1.12.0
```

## Run test
```bash
source env.sh
./test_cpp_ut.sh
./test_python_ut.sh
```

