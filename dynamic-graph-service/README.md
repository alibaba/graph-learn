# 动态图采样服务

## 协议

Apache License 2.0.

## Build Steps
```shell
./build_denpendencies.sh
mkdir -p build && cd build
cmake (-DDEBUG=ON) ..
make -j
make package
```