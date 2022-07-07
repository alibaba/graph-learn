#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
code_src=${script_dir}/hiactor
install_prefix=${script_dir}/build
cores=$(cat < /proc/cpuinfo | grep -c "processor")

cd "${code_src}" && git submodule update --init && \
mkdir -p build && cd build && \
cmake -DCMAKE_CXX_FLAGS="-fPIC" \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -DHiactor_DEMOS=OFF \
  -DHiactor_TESTING=OFF \
  -DHiactor_CXX_DIALECT=gnu++17 \
  -DSeastar_CXX_FLAGS="-DSEASTAR_DEFAULT_ALLOCATOR" \
  .. && \
make -j"${cores}" && make install
