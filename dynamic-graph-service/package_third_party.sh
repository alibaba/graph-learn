#!/bin/bash
#
# A script to package built files of graphlearn third party.

set -eo pipefail

script_dir=$(dirname "$(realpath "$0")")
third_party_dir=${script_dir}/../third_party

pushd "${third_party_dir}"
tar -zcvf dgs_third_party_built.tgz \
  cppkafka/build \
  flatbuffers/build \
  glog/build \
  googletest/build \
  grpc/build \
  hiactor/build \
  rocksdb/build
popd
