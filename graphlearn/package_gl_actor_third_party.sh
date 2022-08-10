#!/bin/bash
#
# A script to package built files of graphlearn-actor third party.

set -eo pipefail

script_dir=$(dirname "$(realpath "$0")")
third_party_dir=${script_dir}/../third_party

pushd "${third_party_dir}"
tar -zcvf gl_actor_third_party_built.tgz \
  hiactor/build \
  glog/build \
  googletest/build \
  grpc/build
popd
