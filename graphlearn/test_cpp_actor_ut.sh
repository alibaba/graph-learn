#!/bin/bash
#
# A script to perform cpp unit tests for graphlearn-actor.

set -eo pipefail

script_dir=$(dirname "$(realpath "$0")")

pushd "${script_dir}"
source env.sh
pushd "${script_dir}/built/bin"
./actor_dag_scheduler_unittest
./batch_generator_unittest
./sharded_graph_store_unittest
popd
popd
