# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cmath import log

""" Packages dependency:
Run the following command first.
For python2.7:
``` shell
  pip install grpcio==1.36.1
  pip install grpcio-tools==1.36
  pip install pyyaml
```
For python3.6
``` shell
  pip3 install grpcio
  pip3 install grpcio-tools
  pip3 install pyyaml
```
Note: default grpcio version is not suppoerted for python2.7.
async api `awit` is not supported for python2.7.

Generate python grpc code:
```
python3 -m grpc_tools.protoc -I../../proto \
--python_out=. \
--grpc_python_out=. ../../proto/coordinator.proto
```

Run:
```
python3 coordinator.py
python3 test_client.py
```
Stop coordinator with KeyboardInterrupt.

"""

import logging
import threading
import os
import argparse
import yaml
from grpc_service import CoordinatorGrpcService
from http_service import CoordinatorHttpService


class Meta(object):
  def __init__(self):
    self._registerd_queries = {}
    self._lock = threading.Lock()
    self._qid = -1

  def _inc_qid(self):
    with self._lock:
      self._qid += 1
    return self._qid

  def register(self, query_id, query):
    for k, v in self._registerd_queries.items():
      if v == query:
        logging.info("Query {} has been registered.".format(k))
        query_id[0] = k
        return False
    query_id[0] = self._inc_qid()
    self._registerd_queries[query_id[0]] = query
    return True


class Coordinator(object):
  def __init__(self, config_dict, grpc_port, http_port):
    assert (type(config_dict) == dict)
    self._meta = Meta()
    self._grpc_service = CoordinatorGrpcService(config_dict, grpc_port)
    self._http_service = CoordinatorHttpService(config_dict, self._grpc_service, self._meta, http_port)

    self._http_server_t = threading.Thread(target=self._http_service.start, daemon=True)
    self._grpc_server_t = threading.Thread(target=self._grpc_service.start, daemon=True)

  def start_and_run(self):
    self._http_server_t.start()
    self._grpc_server_t.start()
    self._http_server_t.join()
    self._grpc_server_t.join()

  def stop(self):
    self._grpc_service.stop()
    self._http_service.stop()


def make_service_config(yaml_config):
  schema_file = yaml_config.get("schema-file", "")
  meta_dir = yaml_config.get("meta-dir", "./coordinator_meta")

  data_loading_yaml_map = yaml_config.get("data-loading", {})
  num_data_loader = data_loading_yaml_map.get("worker-num", 1)

  sampling_yaml_map = yaml_config.get("sampling", {})
  num_sampling_workers = sampling_yaml_map.get("worker-num", 1)
  sampling_actor_local_shard_num = sampling_yaml_map.get("actor-local-shard-num", 1)
  num_sampling_store_partition = sampling_yaml_map.get("store-partitions", 1)
  sampling_store_partition_strategy = sampling_yaml_map.get("store-partition-strategy", "hash")
  serving_worker_partition_strategy = sampling_yaml_map.get("downstream-partition-strategy", "hash")

  serving_yaml_map = yaml_config.get("serving", {})
  num_serving_workers = serving_yaml_map.get("worker-num", 1)
  serving_actor_local_shard_num = serving_yaml_map.get("actor-local-shard-num", 1)
  num_serving_store_partition = serving_yaml_map.get("store-partitions", 1)
  serving_store_partition_strategy = serving_yaml_map.get("store-partition-strategy", "hash")

  kafka_yaml_map = yaml_config.get("kafka", {})

  dl2spl_yaml_map = kafka_yaml_map.get("dl2spl", {})
  dl2spl_kafka_servers = dl2spl_yaml_map.get("servers", ["localhost:9092"])
  dl2spl_kafka_topic = dl2spl_yaml_map.get("topic", "record-batches")
  dl2spl_kafka_partition_num = dl2spl_yaml_map.get("partitions", 1)

  spl2srv_yaml_map = kafka_yaml_map.get("spl2srv", {})
  spl2srv_kafka_servers = spl2srv_yaml_map.get("servers", ["localhost:9092"])
  spl2srv_kafka_topic = spl2srv_yaml_map.get("topic", "sample-batches")
  spl2srv_kafka_partition_num = spl2srv_yaml_map.get("partitions", 1)

  assert dl2spl_kafka_partition_num >= num_sampling_workers
  assert num_sampling_store_partition >= num_sampling_workers
  assert spl2srv_kafka_partition_num >= num_serving_workers

  sampling_store_pids_group = dict()
  for i in range(0, num_sampling_store_partition):
    wid = i % num_sampling_workers
    if wid not in sampling_store_pids_group:
      sampling_store_pids_group[wid] = [i]
    else:
      sampling_store_pids_group[wid].append(i)

  logging.info("---  sampling_store_pids_group  ---")
  for wid, pids in sampling_store_pids_group.items():
    logging.info("worker id: {}, store pids: {}".format(wid, pids))

  serving_store_pids_group = dict()
  for i in range(0, num_serving_store_partition):
    for wid in range(0, num_serving_workers):
      if wid not in serving_store_pids_group:
        serving_store_pids_group[wid] = [i]
      else:
        serving_store_pids_group[wid].append(i)

  logging.info("---  serving_store_pids_group  ---")
  for wid, pids in serving_store_pids_group.items():
    logging.info("worker id: {}, store pids: {}".format(wid, pids))

  sampling_sub_kafka_pids = dict()
  for i in range(0, dl2spl_kafka_partition_num):
    wid = i % num_sampling_workers
    if wid not in sampling_sub_kafka_pids:
      sampling_sub_kafka_pids[wid] = [i]
    else:
      sampling_sub_kafka_pids[wid].append(i)

  logging.info("---  sampling_sub_kafka_pids  ---")
  for wid, pids in sampling_sub_kafka_pids.items():
    logging.info("worker id: {}, kafka pids: {}".format(wid, pids))

  serving_sub_kafka_pids = dict()
  for i in range(0, spl2srv_kafka_partition_num):
    wid = i % num_serving_workers
    if wid not in serving_sub_kafka_pids:
      serving_sub_kafka_pids[wid] = [i]
    else:
      serving_sub_kafka_pids[wid].append(i)

  logging.info("---  serving_sub_kafka_pids  ---")
  for wid, pids in serving_sub_kafka_pids.items():
    logging.info("worker id: {}, kafka pids: {}".format(wid, pids))

  dataloader_pub_partition_vec = [0] * num_sampling_store_partition
  for wid, pids in sampling_store_pids_group.items():
    size = len(sampling_sub_kafka_pids[wid])
    for i in range(len(pids)):
      kafka_pid = sampling_sub_kafka_pids[wid][i % size]
      dataloader_pub_partition_vec[pids[i]] = kafka_pid

  logging.info("---  mapping vector: sampling worker store partition -> dl2spl kafka partition  ---")
  logging.info("{}".format(dataloader_pub_partition_vec))

  sampling_pub_partition_vec = [0] * spl2srv_kafka_partition_num
  for wid, kafka_pids in serving_sub_kafka_pids.items():
    for i in range(len(kafka_pids)):
      sampling_pub_partition_vec[kafka_pids[i]] = wid

  logging.info("---  mapping vector: spl2srv kafka partition -> serving worker id  ---")
  logging.info("{}".format(sampling_pub_partition_vec))

  configs = {
    "schema_file": schema_file,
    "meta_dir": meta_dir,
    "data_loading": {
      "worker_num": num_data_loader,
      "downstream": {
        "kafka": {
          "pub_kafka_servers": dl2spl_kafka_servers,
          "pub_kafka_topic": dl2spl_kafka_topic,
          "pub_kafka_partition_num": dl2spl_kafka_partition_num,
        },
        "partition": {
          "store_partition_strategy": sampling_store_partition_strategy,
          "store_partition_num": num_sampling_store_partition,
          "store_to_kafka_pid_vec": dataloader_pub_partition_vec
        }
      },
    },
    "sampling": {
      "worker_num": num_sampling_workers,
      "num_local_shards": sampling_actor_local_shard_num,
      "store_partition": {
        "partition_strategy": sampling_store_partition_strategy,
        "partition_num": num_sampling_store_partition,
        "managed_pids_group": sampling_store_pids_group,
      },
      "upstream": {
        "sub_kafka_servers": dl2spl_kafka_servers,
        "sub_kafka_topic": dl2spl_kafka_topic,
        "sub_kafka_partition_num": dl2spl_kafka_partition_num,
        "sub_kafka_pids": sampling_sub_kafka_pids,
      },
      "downstream": {
        "kafka": {
          "pub_kafka_servers": spl2srv_kafka_servers,
          "pub_kafka_topic": spl2srv_kafka_topic,
          "pub_kafka_partition_num": spl2srv_kafka_partition_num,
        },
        "partition": {
          "worker_partition_strategy": serving_worker_partition_strategy,
          "worker_partition_num": num_serving_workers,
          "kafka_to_worker_pid_vec": sampling_pub_partition_vec
        }
      }
    },
    "serving": {
      "worker_num": num_serving_workers,
      "num_local_shards": serving_actor_local_shard_num,
      "store_partition": {
        "partition_strategy": serving_store_partition_strategy,
        "partition_num": num_serving_store_partition,
        "managed_pids_group": serving_store_pids_group,
      },
      "upstream": {
        "sub_kafka_servers": spl2srv_kafka_servers,
        "sub_kafka_topic": spl2srv_kafka_topic,
        "sub_kafka_partition_num": spl2srv_kafka_partition_num,
        "sub_kafka_pids": serving_sub_kafka_pids,
      },
    }
  }

  return configs


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  os.system("python3 -m grpc_tools.protoc -I../../proto \
            --python_out=. \
            --grpc_python_out=. ../../proto/coordinator.proto")
  parser = argparse.ArgumentParser(description='Coordinator Arguments.')
  parser.add_argument('--config-file', action="store", dest="config_file",
                      help="yaml config file of coordinator")
  args = parser.parse_args()
  if args.config_file is None:
    yaml_config = {}
  else:
    yaml_config = yaml.load(open(args.config_file), Loader=yaml.FullLoader)
  configs = make_service_config(yaml_config)
  coord = Coordinator(configs, 50051, 8080)
  try:
    coord.start_and_run()
  except KeyboardInterrupt:
    pass
  logging.info("Stopping Coordinator...\n")
  coord.stop()
  logging.info("Coordinator Stopped.\n")
