#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
code_src=${script_dir}/rocksdb
install_prefix=${script_dir}/build
cores=$(cat < /proc/cpuinfo | grep -c "processor")

cd "${code_src}" && \
mkdir -p build && cd build && \
cmake -DCMAKE_CXX_FLAGS="-fPIC" \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCKSDB_BUILD_SHARED=OFF \
  -DUSE_RTTI=1 \
  -DWITH_SNAPPY=ON \
  -DWITH_LZ4=ON \
  -DWITH_ZLIB=ON \
  -DWITH_ZSTD=ON \
  -DWITH_BENCHMARK_TOOLS=OFF \
  -DWITH_CORE_TOOLS=OFF \
  -DWITH_TOOLS=OFF \
  .. && \
make -j"${cores}" && make install
