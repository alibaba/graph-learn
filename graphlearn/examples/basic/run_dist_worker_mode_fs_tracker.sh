#!/usr/bin/env bash
HERE=$(cd "$(dirname "$0")";pwd)

rm -rf tracker
mkdir -p tracker

# Only generating data when ./data folder is not existed.
# If `gen_test_data.py` is modified, then delete the data folder first.
if [ ! -d "$HERE/data" ]; then
  mkdir -p $HERE/data
  python $HERE/gen_test_data.py
fi

# Start a graphlearn cluster with 2 workers(processes).
python $HERE/test_dist_worker_mode_fs_tracker.py\
  --task_index=0 --task_count=2 --tracker="tracker" &
sleep 1
python $HERE/test_dist_worker_mode_fs_tracker.py \
  --task_index=1 --task_count=2 --tracker="tracker"