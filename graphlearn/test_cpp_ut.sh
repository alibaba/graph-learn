#!/bin/bash

set -eo pipefail

script_dir=$(dirname "$(realpath "$0")")

pushd "${script_dir}"
source env.sh
cd /built/bin
for i in *_unittest
  do ./"$i"
done
popd
