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

import logging
import os
import threading
from enum import IntEnum

from checkpoint import SubServiceCheckpointManager


class WorkerState(IntEnum):
  TERMINATED = 0
  REGISTERED = 1
  STARTED = 2


class SubServiceStateManager(object):
  def __init__(self, num_workers):
    self.name = ""
    self.num_workers = num_workers
    self._ip_addrs = ["" for _ in range(num_workers)]
    self._states = [WorkerState.TERMINATED for _ in range(num_workers)]
    self._lock = threading.Lock()

  def check_is_terminated(self, worker_id):
    with self._lock:
      return int(self._states[worker_id]) == int(WorkerState.TERMINATED)

  def check_is_registered(self, worker_id):
    with self._lock:
      return int(self._states[worker_id]) == int(WorkerState.REGISTERED)

  def check_is_started(self, worker_id):
    with self._lock:
      return int(self._states[worker_id]) == int(WorkerState.STARTED)

  def check_all_have_registered(self):
    with self._lock:
      return self._check_states_of_all(WorkerState.REGISTERED)

  def check_all_have_started(self):
    with self._lock:
      return self._check_states_of_all(WorkerState.STARTED)

  def register_worker(self, worker_id, worker_ip, independent=True):
    if (worker_id < 0) or (worker_id >= self.num_workers):
      logging.warning("Registration request of {}-{} is invalid, ignore!"
                      .format(self.name, worker_id))
      return False
    with self._lock:
      if (not independent) and (self._check_states_of_all(WorkerState.REGISTERED)):
        self._ip_addrs = ["" for _ in range(self.num_workers)]
        self._states = [WorkerState.TERMINATED for _ in range(self.num_workers)]
        logging.info("Registration request of {}-{} is abnormal, "
                     "try to restart all workers of {} ..."
                     .format(self.name, worker_id, self.name))
      self._ip_addrs[worker_id] = worker_ip
      self._states[worker_id] = WorkerState.REGISTERED
      logging.info("{}-{} is registered with ip {}"
                   .format(self.name, worker_id, worker_ip))
    return True

  def set_started(self, worker_id):
    with self._lock:
      self._states[worker_id] = WorkerState.STARTED
      logging.info("{}-{} is started".format(self.name, worker_id))

  def _check_states_of_all(self, expected_state):
    for s in self._states:
      if int(s) < int(expected_state):
        return False
    return True


class SamplingStateManager(SubServiceStateManager):
  def __init__(self, sampling_configs, meta_root_dir):
    num_workers = sampling_configs.get("worker_num")
    super().__init__(num_workers)
    self.name = "SamplingWorker"
    self.checkpoint_manager = SubServiceCheckpointManager(
      worker_num=num_workers,
      meta_dir=os.path.join(meta_root_dir, "checkpoint", self.name),
      kafka_pids_group=sampling_configs.get("upstream").get("sub_kafka_pids"),
      store_pids_group=sampling_configs.get("store_partition").get("managed_pids_group"),
      independent=False
    )
    self.info = sampling_configs
    logging.info("Configure service with {} Sampling Workers.".format(num_workers))

  def get_composed_ip_list(self):
    composed_ip_list = []
    for i in range(self.num_workers):
      composed_ip_list.append("{} {} {}".format(i, self.info.get("num_local_shards"), self._ip_addrs[i]))
    return composed_ip_list


class ServingStateManager(SubServiceStateManager):
  def __init__(self, serving_configs, meta_root_dir):
    num_workers = serving_configs.get("worker_num")
    super().__init__(num_workers)
    self.name = "ServingWorker"
    self.checkpoint_manager = SubServiceCheckpointManager(
      worker_num=num_workers,
      meta_dir=os.path.join(meta_root_dir, "checkpoint", self.name),
      kafka_pids_group=serving_configs.get("upstream").get("sub_kafka_pids"),
      store_pids_group=serving_configs.get("store_partition").get("managed_pids_group"),
      independent=True
    )
    self.info = serving_configs
    logging.info("Configure service with {} Serving Workers.".format(num_workers))
