#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
code_src=${script_dir}/flatbuffers
install_prefix=${script_dir}/build
cores=$(cat < /proc/cpuinfo | grep -c "processor")

cd "${code_src}" && \
cmake -G "Unix Makefiles" \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLATBUFFERS_CXX_FLAGS="-fPIC" \
  -DFLATBUFFERS_BUILD_TESTS=OFF && \
make -j"${cores}" && make install