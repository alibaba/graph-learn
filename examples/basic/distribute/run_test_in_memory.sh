#!/usr/bin/env bash
HERE=$(cd "$(dirname "$0")";pwd)

rm -rf $HERE/tracker
mkdir -p $HERE/tracker

if [ ! -d "$HERE/data" ]; then
  mkdir -p $HERE/data
  python $HERE/gen_test_data.py
fi

python $HERE/test_in_memory.py \
  --task_index=0 --task_count=2 --tracker="$HERE/tracker" &
sleep 1
python $HERE/test_in_memory.py \
  --task_index=1 --task_count=2 --tracker="$HERE/tracker"

# sleep 5
rm -rf $HERE/tracker
python $HERE/test_in_memory.py \
  --task_index=0 --hosts="127.0.0.1:8888,127.0.0.1:8889" &
sleep 1
python $HERE/test_in_memory.py \
  --task_index=1 --hosts="127.0.0.1:8888,127.0.0.1:8889"