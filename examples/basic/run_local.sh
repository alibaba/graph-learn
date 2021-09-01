#!/usr/bin/env bash
HERE=$(cd "$(dirname "$0")";pwd)

# Only generating data when ./data folder is not existed.
# If `gen_test_data.py` is modified, then delete the data folder first.
if [ ! -d "$HERE/data" ]; then
  mkdir -p $HERE/data
  python $HERE/gen_test_data.py
fi

python $HERE/test_local.py
