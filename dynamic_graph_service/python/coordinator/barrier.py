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
import pickle
import threading
import time

from kafka import KafkaConsumer
from kafka.structs import TopicPartition


def check_barrier_offsets(barrier_offsets, ready_offsets):
  for pid, offset in barrier_offsets.items():
    if pid not in ready_offsets.keys():
      return False
    if ready_offsets[pid] < offset:
      return False
  return True


class KafkaOffsetsFetcher(object):
  def __init__(self, brokers, topic, partitions):
    self._consumer_client = KafkaConsumer(
      bootstrap_servers=brokers,
      group_id="{}-offsets-fetcher".format(topic),
    )
    self._topic_partitions = [TopicPartition(topic, p) for p in range(partitions)]

    # pause this consumer as we never poll messages.
    self._consumer_client.assign([])
    self._consumer_client.pause()
    pass

  def query_current_offsets(self):
    end_offsets = self._consumer_client.end_offsets(self._topic_partitions)
    pid_to_offset_dict = {}
    for p, offset in end_offsets.items():
      pid_to_offset_dict[p.partition] = offset - 1
    return pid_to_offset_dict


class SubServiceBarrierState(object):
  def __init__(self, checkpoint_mgr):
    self._checkpoint_mgr = checkpoint_mgr
    self._barrier_offsets_dict = {}
    self._lock = threading.Lock()

  def add_barrier(self, barrier_name, barrier_offsets):
    with self._lock:
      self._barrier_offsets_dict[barrier_name] = barrier_offsets

  def remove_barrier(self, barrier_name):
    with self._lock:
      self._barrier_offsets_dict.pop(barrier_name)

  def contains_barrier(self, barrier_name):
    with self._lock:
      return barrier_name in self._barrier_offsets_dict.keys()

  def collect_finished_barriers(self):
    with self._lock:
      if len(self._barrier_offsets_dict) == 0:
        return []
      finished_barriers = []
      ready_offsets = self._checkpoint_mgr.get_all_ready_offsets()
      for name, barrier_offsets in self._barrier_offsets_dict.items():
        if check_barrier_offsets(barrier_offsets, ready_offsets):
          finished_barriers.append(name)
      return finished_barriers


class GlobalBarrierMonitor(object):
  def __init__(self, meta_root_dir, dl2spl_fetcher, spl2srv_fetcher, sampling_checkpoint_mgr, serving_checkpoint_mgr):
    self._barrier_meta_dir = os.path.join(meta_root_dir, "barrier")
    self._dl2spl_fetcher = dl2spl_fetcher
    self._spl2srv_fetcher = spl2srv_fetcher
    self._sampling_barrier_state = SubServiceBarrierState(sampling_checkpoint_mgr)
    self._serving_barrier_state = SubServiceBarrierState(serving_checkpoint_mgr)
    self._pending_barriers = {}
    self._ready_barriers = set()

    # restore existed barriers
    self.__restore_barriers()

    # scheduled task executor for barrier checking every 5 seconds
    self._checking_t = threading.Thread(
      target=self.__check_and_switch_barrier_states,
      daemon=True
    )
    self._stopped = False

  def start(self):
    self._checking_t.start()

  def stop(self):
    self._stopped = True
    self._checking_t.join()

  def set_barrier_from_dataloader(self, barrier_name, dl_count, dl_id):
    if barrier_name not in self._pending_barriers.keys():
      self._pending_barriers[barrier_name] = set()
    self._pending_barriers[barrier_name].add(dl_id)
    if len(self._pending_barriers[barrier_name]) >= dl_count:
      dl2spl_offsets = self._dl2spl_fetcher.query_current_offsets()
      self.__persist_barrier(barrier_name, dl2spl_offsets)
      self._sampling_barrier_state.add_barrier(barrier_name, dl2spl_offsets)
      self._pending_barriers.pop(barrier_name)
      logging.info("Global barrier {} has been set.".format(barrier_name))

  def check_status(self, barrier_name):
    if barrier_name in self._ready_barriers:
      return "READY"
    elif self._serving_barrier_state.contains_barrier(barrier_name):
      return "SAMPLED"
    elif self._sampling_barrier_state.contains_barrier(barrier_name):
      return "PRODUCED"
    else:
      return "NOT_SET"

  def check_existed(self, barrier_name):
    return (barrier_name in self._ready_barriers) or \
           (self._serving_barrier_state.contains_barrier(barrier_name)) or \
           (self._sampling_barrier_state.contains_barrier(barrier_name))

  def __persist_barrier(self, barrier_name, barrier_offsets):
    f = os.path.join(self._barrier_meta_dir, barrier_name)
    with open(f, 'wb') as out_data:
      pickle.dump(barrier_offsets, out_data, pickle.HIGHEST_PROTOCOL)

  def __restore_barriers(self):
    # We just need to restore barriers for sampling, as all of them will be moved to serving stage finally.
    if not os.path.exists(self._barrier_meta_dir):
      os.makedirs(self._barrier_meta_dir)
    for filename in os.listdir(self._barrier_meta_dir):
      f = os.path.join(self._barrier_meta_dir, filename)
      with open(f, 'rb') as in_data:
        offsets = pickle.load(in_data)
      self._sampling_barrier_state.add_barrier(filename, offsets)

  def __check_and_switch_barrier_states(self):
    while not self._stopped:
      # execute every 5 seconds
      time.sleep(5)
      sampling_finished = self._sampling_barrier_state.collect_finished_barriers()
      if len(sampling_finished) > 0:
        spl2srv_offsets = self._spl2srv_fetcher.query_current_offsets()
        for name in sampling_finished:
          self._sampling_barrier_state.remove_barrier(name)
          self._serving_barrier_state.add_barrier(name, spl2srv_offsets)
      serving_finished = self._serving_barrier_state.collect_finished_barriers()
      if len(serving_finished) > 0:
        for name in serving_finished:
          self._serving_barrier_state.remove_barrier(name)
          self._ready_barriers.add(name)
          logging.info("Global barrier {} is ready.".format(name))
