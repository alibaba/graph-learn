#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
code_src=${script_dir}/glog
install_prefix=${script_dir}/build
cores=$(cat < /proc/cpuinfo | grep -c "processor")

cd "${code_src}" && \
mkdir -p build && cd build && \
cmake -DCMAKE_CXX_FLAGS="-fPIC" \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTING=OFF \
  .. && \
make -j"${cores}" && make install && \
cd "${code_src}" && rm -rf build
