#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
code_src=${script_dir}/cppkafka
install_prefix=${script_dir}/build
cores=$(cat < /proc/cpuinfo | grep -c "processor")

cd "${code_src}" && mkdir -p build && cd build && \
cmake -DCMAKE_CXX_FLAGS="-fPIC" \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -DCPPKAFKA_BUILD_SHARED=0 \
  -DCPPKAFKA_DISABLE_TESTS=ON \
  -DCPPKAFKA_DISABLE_EXAMPLES=ON \
  .. && \
make -j"${cores}" && make install
