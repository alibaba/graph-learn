#!/bin/bash
#
# A script to install dependencies and third parties for graphlearn.

set -eo pipefail

gl_root_dir=$(dirname "$(realpath "$0")")
third_party_dir=${gl_root_dir}/../third_party

build_hiactor=false

for i in "$@"; do
  case $i in
    --build-hiactor)
      build_hiactor=true
      shift # past argument with no value
      ;;
    --*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

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
  libopenblas-dev
  libgflags-dev
  libssl-dev
  libc-ares-dev
  zlib1g-dev
)

if [ "${build_hiactor}" = true ] ; then
  debian_packages=(
    "${debian_packages[@]}"
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
    liblz4-dev
    libsctp-dev
    systemtap-sdt-dev
    libtool
    libyaml-cpp-dev
    stow
    libfmt-dev
    diffutils
    valgrind
  )
fi

# installing dgs system dependencies
if [ "$ID" = "ubuntu" ] || [ "$ID" = "debian" ]; then
  apt-get update
  apt-get -y install "${debian_packages[@]}"
else
  echo "Your system ($ID) is not supported by this script. Please install dependencies manually or build in ubuntu env."
  exit 1
fi

## installing submodules

# hiactor
if [ "${build_hiactor}" = true ] ; then
  echo "-- installing hiactor ..."
  if [ ! -f "${third_party_dir}/hiactor/build/include/hiactor/core/actor-template.hh" ]; then
    pushd "${third_party_dir}/hiactor"
    git submodule update --init hiactor
    /bin/bash build.sh
    popd
  fi
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

# pybind11
echo "-- preparing pybind11 ..."
pushd "${third_party_dir}/pybind11"
git submodule update --init pybind11
popd
