#!/bin/bash

set -eo pipefail

script_dir=$(dirname "$(realpath "$0")")
JAVA_LIB=${JAVA_HOME}/jre/lib/amd64/server/
LD_LIBRARY_PATH=${script_dir}/built/lib:${JAVA_LIB}:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
