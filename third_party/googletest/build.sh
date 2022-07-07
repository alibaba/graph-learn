#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
code_src=${script_dir}/googletest
install_prefix=${script_dir}/build
cores=$(cat < /proc/cpuinfo | grep -c "processor")

cd "${code_src}" && mkdir -p build && cd build && \
cmake -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_INSTALL_PREFIX="${install_prefix}" .. && \
make -j"${cores}" && make install
