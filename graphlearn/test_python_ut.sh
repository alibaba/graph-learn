#!/bin/bash

set -eo pipefail

usage() {
  echo "Usage: $0 [-p <python | python3>" 1>&2; exit 1;
}

PYTHON=python
while getopts ":p:" o; do
    case "${o}" in
        p)
            PYTHON=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${PYTHON}" ]; then
    usage
fi

script_dir=$(dirname "$(realpath "$0")")

pushd "${script_dir}"/python
for file in */test_*.py
do
  echo "$file"
  ${PYTHON} "$file"
done
popd
