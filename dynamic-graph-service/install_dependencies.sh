#!/bin/bash
#
# A script to install dependencies and third parties for dynamic graph service.

set -e

dgs_root_dir=$(dirname "$(realpath "$0")")
third_party_dir=${dgs_root_dir}/../third_party

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
  # jdk
  openjdk-8-jdk
  # project common dependencies
  libyaml-cpp-dev
  librdkafka-dev
  libgflags-dev
  libssl-dev
  libjemalloc-dev
  libc-ares-dev
  zlib1g-dev
  libcurl4-openssl-dev
  # hiactor dependencies
  ragel
  libhwloc-dev
  libnuma-dev
  libpciaccess-dev
  libcrypto++-dev
  libboost-all-dev
  libxml2-dev
  xfslibs-dev
  libgnutls28-dev
  libsctp-dev
  systemtap-sdt-dev
  libtool
  stow
  libfmt-dev
  diffutils
  valgrind
  # rocksdb dependencies
  libsnappy-dev
  libbz2-dev
  liblz4-dev
  libzstd-dev
)

# installing dgs system dependencies
if [ "$ID" = "ubuntu" ] || [ "$ID" = "debian" ]; then
  sudo apt-get -y install "${debian_packages[@]}"
else
  echo "Your system ($ID) is not supported by this script. Please install dependencies manually or build in ubuntu env."
  exit 1
fi

## installing submodules

# hiactor
echo "-- installing hiactor ..."
if [ ! -f "${third_party_dir}/hiactor/build/include/hiactor/core/actor-template.hh" ]; then
  pushd "${third_party_dir}/hiactor"
  git submodule update --init hiactor
  /bin/bash build.sh
  popd
fi

# cppkafka
echo "-- installing cppkafka ..."
if [ ! -f "${third_party_dir}/cppkafka/build/include/cppkafka/cppkafka.h" ]; then
  pushd "${third_party_dir}/cppkafka"
  git submodule update --init cppkafka
  /bin/bash build.sh
  popd
fi

# flatbuffers
echo "-- installing flatbuffers ..."
if [ ! -f "${third_party_dir}/flatbuffers/build/include/flatbuffers/flatbuffers.h" ]; then
  pushd "${third_party_dir}/flatbuffers"
  git submodule update --init flatbuffers
  /bin/bash build.sh
  popd
fi

# glog
echo "-- installing glog ..."
if [ ! -f "${third_party_dir}/glog/build/include/glog/logging.h" ]; then
  pushd "${third_party_dir}/glog"
  git submodule update --init glog
  /bin/bash build.sh
  popd
fi

# googletest
echo "-- installing googletest ..."
if [ ! -f "${third_party_dir}/googletest/build/include/gtest/gtest.h" ]; then
  pushd "${third_party_dir}/googletest"
  git submodule update --init googletest
  /bin/bash build.sh
  popd
fi

# grpc
echo "-- installing grpc ..."
if [ ! -f "${third_party_dir}/grpc/build/include/grpc++/grpc++.h" ]; then
  pushd "${third_party_dir}/grpc"
  git submodule update --init grpc
  /bin/bash build.sh
  popd
fi

# rocksdb
echo "-- installing rocksdb ..."
if [ ! -f "${third_party_dir}/rocksdb/build/include/rocksdb/db.h" ]; then
  pushd "${third_party_dir}/rocksdb"
  git submodule update --init rocksdb
  /bin/bash build.sh
  popd
fi
