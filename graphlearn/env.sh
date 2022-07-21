#!/bin/bash

set -eo pipefail

script_dir=$(dirname "$(realpath "$0")")
JAVA_LIB=${JAVA_HOME}/jre/lib/amd64/server/
SYS_LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64
LD_LIBRARY_PATH=$SYS_LD_LIBRARY_PATH:${script_dir}/built/lib:${JAVA_LIB}:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
