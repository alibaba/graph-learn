#!/bin/bash
#
# A script to perform cpp unit tests for graphlearn.

set -eo pipefail

script_dir=$(dirname "$(realpath "$0")")

pushd "${script_dir}"
source env.sh
cd "${script_dir}/built/bin"
for i in *_unittest
  do ./"$i"
done
popd
