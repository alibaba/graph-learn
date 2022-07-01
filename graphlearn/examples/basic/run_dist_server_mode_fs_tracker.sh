#!/usr/bin/env bash
HERE=$(cd "$(dirname "$0")";pwd)

rm -rf $HERE/tracker
mkdir -p $HERE/tracker

# Only generating data when ./data folder is not existed.
# If `gen_test_data.py` is modified, then delete the data folder first.
if [ ! -d "$HERE/data" ]; then
  mkdir -p $HERE/data
  python $HERE/gen_test_data.py
fi

# Start a graphlearn cluster with 2 servers(processes) and 3 clients(processes).
python $HERE/test_dist_server_mode_fs_tracker.py \
  --server_count=2 --client_count=3 --tracker=$HERE/tracker --job_name="server" --task_index=0 &
sleep 1
python $HERE/test_dist_server_mode_fs_tracker.py \
  --server_count=2 --client_count=3 --tracker=$HERE/tracker --job_name="server" --task_index=1 &
sleep 1
python $HERE/test_dist_server_mode_fs_tracker.py \
  --server_count=2 --client_count=3 --tracker=$HERE/tracker --job_name="client" --task_index=0 &
sleep 1
python $HERE/test_dist_server_mode_fs_tracker.py \
  --server_count=2 --client_count=3 --tracker=$HERE/tracker --job_name="client" --task_index=1 &
sleep 1
python $HERE/test_dist_server_mode_fs_tracker.py \
  --server_count=2 --client_count=3 --tracker=$HERE/tracker --job_name="client" --task_index=2