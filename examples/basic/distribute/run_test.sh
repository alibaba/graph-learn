#!/usr/bin/env bash
HERE=$(cd "$(dirname "$0")";pwd)

rm -rf tracker
mkdir -p tracker

if [ ! -d "$HERE/data" ]; then
  mkdir -p $HERE/data
  python $HERE/gen_test_data.py
fi

python $HERE/test.py \
  --cluster="{\"client_count\": 2, \"tracker\": \"tracker\", \"server\": \"127.0.0.1:8888,127.0.0.1:8889\"}" \
  --job_name="server" --task_index=0 &
sleep 1
python $HERE/test.py \
  --cluster="{\"client_count\": 2, \"tracker\": \"tracker\", \"server\": \"127.0.0.1:8888,127.0.0.1:8889\"}" \
  --job_name="server" --task_index=1 &
sleep 1
python $HERE/test.py \
  --cluster="{\"client_count\": 2, \"tracker\": \"tracker\", \"server\": \"127.0.0.1:8888,127.0.0.1:8889\"}" \
  --job_name="client" --task_index=0 &
sleep 1
python $HERE/test.py \
  --cluster="{\"client_count\": 2, \"tracker\": \"tracker\", \"server\": \"127.0.0.1:8888,127.0.0.1:8889\"}" \
  --job_name="client" --task_index=1

# test iterate with server cout < client cout.
sleep 5
rm -rf tracker
mkdir -p tracker
python $HERE/test_iterate.py \
  --cluster="{\"client_count\": 2, \"tracker\": \"tracker\", \"server_count\": 1}" \
  --job_name="server" --task_index=0 --mode=1&
sleep 1
python $HERE/test_iterate.py \
  --cluster="{\"client_count\": 2, \"tracker\": \"tracker\", \"server_count\": 1}" \
  --job_name="client" --task_index=0 --mode=1&
sleep 1
python $HERE/test_iterate.py \
  --cluster="{\"client_count\": 2, \"tracker\": \"tracker\", \"server_count\": 1}" \
  --job_name="client" --task_index=1 --mode=1
