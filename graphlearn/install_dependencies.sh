#!/bin/bash
#
# A script to install dependencies and third parties for graphlearn.

set -e

gl_root_dir=$(dirname "$(realpath "$0")")
third_party_dir=${gl_root_dir}/../third_party

# os-release may be missing in container environment by default.
if [ -f "/etc/os-release" ]; then
  . /etc/os-release
elif [ -f "/etc/arch-release" ]; then
  export ID=arch
else
  echo "/etc/os-release missing."
  exit 1
fi

debian_packages=(
  # basic
  gcc
  g++
  make
  cmake
  pkg-config
  # python
  python3
  python3-pip
  # project common dependencies
  libopenblas-dev
  libgflags-dev
  libssl-dev
  libc-ares-dev
  zlib1g-dev
)

# installing dgs system dependencies
if [ "$ID" = "ubuntu" ] || [ "$ID" = "debian" ]; then
  sudo apt-get -y install "${debian_packages[@]}"
else
  echo "Your system ($ID) is not supported by this script. Please install dependencies manually or build in ubuntu env."
  exit 1
fi

## installing submodules

# glog
echo "-- installing glog ..."
if [ ! -f "${third_party_dir}/glog/build/include/glog/logging.h" ]; then
  pushd "${third_party_dir}/glog"
  /bin/bash build.sh
  popd
fi

# googletest
echo "-- installing googletest ..."
if [ ! -f "${third_party_dir}/googletest/build/include/gtest/gtest.h" ]; then
  pushd "${third_party_dir}/googletest"
  /bin/bash build.sh
  popd
fi

# grpc
echo "-- installing grpc ..."
if [ ! -f "${third_party_dir}/grpc/build/include/grpc++/grpc++.h" ]; then
  pushd "${third_party_dir}/grpc"
  /bin/bash build.sh
  popd
fi
