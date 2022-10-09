#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
code_src=${script_dir}/grpc
install_prefix=${script_dir}/build
cores=$(cat < /proc/cpuinfo | grep -c "processor")

cd "${code_src}" && git submodule update --init third_party/protobuf third_party/abseil-cpp third_party/re2 third_party/cares && \
mkdir -p cmake/build && cd cmake/build && \
cmake -DCMAKE_CXX_FLAGS="-fPIC" \
  -DCMAKE_BUILD_TYPE=Release \
  -DgRPC_INSTALL=ON \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -DgRPC_BUILD_TESTS=OFF \
  -DgRPC_SSL_PROVIDER=package \
  -DgRPC_ZLIB_PROVIDER=package \
  ../.. && \
make -j"${cores}" && make install &&
cp -r "${code_src}"/third_party/abseil-cpp/absl "${install_prefix}"/include
cp -r "${code_src}"/third_party/cares/cares "${install_prefix}"/include