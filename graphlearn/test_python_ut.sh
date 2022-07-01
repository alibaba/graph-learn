#!/usr/bin/env bash
usage() { echo "Usage: $0 [-p <python | python3>" 1>&2; exit 1; }
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

for file in $(ls -ld $(find ./graphlearn/python))
do
  if [[ $file == */test_*.py ]]
  then
    echo $file
    ${PYTHON} $file
  fi
done
