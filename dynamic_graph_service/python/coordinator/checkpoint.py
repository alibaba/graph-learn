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
from enum import IntEnum


class KafkaPartitionOffsetInfo(object):
  def __init__(self, pid, ready_offset):
    self.pid = pid
    self.ready_offset = ready_offset


class StorePartitionBackupInfo(object):
  def __init__(self, pid, valid, vertex_bid, edge_bid):
    self.pid = pid
    self.valid = valid
    self.vertex_bid = vertex_bid
    self.edge_bid = edge_bid


class SubsPartitionBackupInfo(object):
  def __init__(self, pid, valid, bid):
    self.pid = pid
    self.valid = valid
    self.bid = bid


class WorkerCheckpoint(object):
  def __init__(self, kafka_offsets, store_backups, subs_backups):
    self.kafka_offsets = kafka_offsets
    self.store_backups = store_backups
    self.subs_backups = subs_backups


class WorkerCheckpointManager(object):
  def __init__(self, meta_dir, kafka_pids, store_pids):
    self._meta_dir = meta_dir
    self._checkpoints = {}

    # set init checkpoint state
    init_kafka_offsets = [KafkaPartitionOffsetInfo(pid, -1) for pid in kafka_pids]
    init_store_backups = [StorePartitionBackupInfo(pid, False, 0, 0) for pid in store_pids]
    init_subs_backups = [SubsPartitionBackupInfo(pid, False, 0) for pid in store_pids]
    self._checkpoints[0] = WorkerCheckpoint(init_kafka_offsets, init_store_backups, init_subs_backups)

    # restore checkpoints from file
    if not os.path.exists(self._meta_dir):
      os.makedirs(self._meta_dir)
    for filename in os.listdir(self._meta_dir):
      f = os.path.join(self._meta_dir, filename)
      if os.path.isfile(f):
        checkpoint_id = int(filename)
        with open(f, 'rb') as in_data:
          checkpoint_info = pickle.load(in_data)
        self._checkpoints[checkpoint_id] = checkpoint_info

    # update available checkpoint id
    self._available_checkpoint_id = max(self._checkpoints.keys()) + 1

  def add_checkpoint(self, checkpoint_info):
    checkpoint_id = self._available_checkpoint_id
    # persist checkpoint info
    f = os.path.join(self._meta_dir, str(checkpoint_id))
    with open(f, 'wb') as out_data:
      pickle.dump(checkpoint_info, out_data, pickle.HIGHEST_PROTOCOL)
    self._checkpoints[checkpoint_id] = checkpoint_info
    self._available_checkpoint_id += 1

  def purge_old_checkpoints(self, keep_num):
    init_checkpoint = self._checkpoints[0]
    self._checkpoints.pop(0)
    while len(self._checkpoints) > keep_num:
      oldest_id = min(self._checkpoints.keys())
      self.__delete_checkpoint(oldest_id)
    self._checkpoints[0] = init_checkpoint

  def get_latest_checkpoint_id(self):
    return max(self._checkpoints.keys())

  def get_ready_offsets(self):
    latest_cp = self._checkpoints[self.get_latest_checkpoint_id()]
    ready_offset_dict = {}
    for kafka_offset in latest_cp.kafka_offsets:
      ready_offset_dict[kafka_offset.pid] = kafka_offset.ready_offset
    return ready_offset_dict

  def set_pb_with_latest_checkpoint(self, pb):
    latest_cp = self._checkpoints[self.get_latest_checkpoint_id()]
    for kafka_offset in latest_cp.kafka_offsets:
      entry = pb.sub_kafka_offsets.add()
      entry.pid = kafka_offset.pid
      entry.ready_offset = kafka_offset.ready_offset
    for store_backup in latest_cp.store_backups:
      entry = pb.sample_store_backups.add()
      entry.pid = store_backup.pid
      entry.valid = store_backup.valid
      entry.vertex_bid = store_backup.vertex_bid
      entry.edge_bid = store_backup.edge_bid
    for subs_backup in latest_cp.subs_backups:
      entry = pb.subs_table_backups.add()
      entry.pid = subs_backup.pid
      entry.valid = subs_backup.valid
      entry.bid = subs_backup.bid

  def __delete_checkpoint(self, checkpoint_id):
    assert checkpoint_id in self._checkpoints.keys()
    f = os.path.join(self._meta_dir, str(checkpoint_id))
    if os.path.exists(f):
      os.remove(f)
    self._checkpoints.pop(checkpoint_id)


class CheckpointState(IntEnum):
  WAITING = 0  # waiting for new checkpoint request
  INCOMING = 1  # there is an incoming checkpoint request but worker has not been notified
  DOING = 2  # worker has been notified to do backups


class SubServiceCheckpointManager(object):
  def __init__(self, worker_num, meta_dir, kafka_pids_group, store_pids_group, independent=False):
    self._worker_num = worker_num
    self._independent_mode = independent
    self._worker_cp_managers = [
      WorkerCheckpointManager(
        os.path.join(meta_dir, str(i)),
        kafka_pids_group[i],
        store_pids_group[i]
      ) for i in range(worker_num)
    ]
    self._ready_offset_dicts = [{} for _ in range(worker_num)]
    self._checkpoint_states = [CheckpointState.WAITING for _ in range(worker_num)]
    self._pending_ready_offsets = []
    self._pending_checkpoints = {}
    self._lock = threading.Lock()

  def update_kafka_ready_offsets(self, worker_id, pb):
    with self._lock:
      ready_offset_dict = self._ready_offset_dicts[worker_id]
      for tup in pb:
        ready_offset_dict[tup.pid] = max(tup.ready_offset, ready_offset_dict[tup.pid])

  def start_checkpointing(self):
    with self._lock:
      if self.__check_all_waiting():
        self._checkpoint_states = [CheckpointState.INCOMING for _ in range(self._worker_num)]
        self._pending_ready_offsets = self._ready_offset_dicts.copy()
        return True
      else:
        logging.warning("The previous checkpoint has not been finished, try it later.")
        return False

  def check_for_backup(self, worker_id):
    with self._lock:
      if self._checkpoint_states[worker_id] is CheckpointState.INCOMING:
        self._checkpoint_states[worker_id] = CheckpointState.DOING
        return True
      return False

  def set_init_pb_with_latest_checkpoint(self, worker_id, pb):
    with self._lock:
      worker_cp_manager = self._worker_cp_managers[worker_id]
      worker_cp_manager.set_pb_with_latest_checkpoint(pb)
      # if worker is restarting, the ready offsets state should roll back to last checkpoint.
      self._ready_offset_dicts[worker_id] = worker_cp_manager.get_ready_offsets()
      if self._independent_mode:
        # independent mode (serving worker): reset state of worker itself.
        self._checkpoint_states[worker_id] = CheckpointState.WAITING
      else:
        # dependent mode (sampling worker): reset state of all workers.
        # if prev checkpoint is not finished, drop it.
        self.__reset_all_checkpoint_status()

  def report_worker_backup_with_pb(self, worker_id, store_backups_pb, subs_backups_pb):
    with self._lock:
      if self._checkpoint_states[worker_id] is not CheckpointState.DOING:
        # invalid worker backup
        return False

      ready_offsets = []
      for pid, ready_offset in self._pending_ready_offsets[worker_id].items():
        ready_offsets.append(KafkaPartitionOffsetInfo(pid, ready_offset))
      store_backups = []
      for tup in store_backups_pb:
        store_backups.append(StorePartitionBackupInfo(tup.pid, tup.valid, tup.vertex_bid, tup.edge_bid))
      subs_backups = []
      for tup in subs_backups_pb:
        subs_backups.append(SubsPartitionBackupInfo(tup.pid, tup.valid, tup.bid))
      worker_cp = WorkerCheckpoint(ready_offsets, store_backups, subs_backups)

      if self._independent_mode:
        # independent mode (serving worker): update directly and no need to wait others.
        self._worker_cp_managers[worker_id].add_checkpoint(worker_cp)
        self._checkpoint_states[worker_id] = CheckpointState.WAITING
      else:
        # dependent mode (sampling worker): caching worker checkpoint and wait others to finish.
        self._pending_checkpoints[worker_id] = worker_cp
        if len(self._pending_checkpoints) == self._worker_num:
          for wid, pending_cp in self._pending_checkpoints.items():
            self._worker_cp_managers[wid].add_checkpoint(pending_cp)
          self.__reset_all_checkpoint_status()
      return True

  def purge_old_checkpoints(self, keep_num):
    with self._lock:
      for worker_cp_manager in self._worker_cp_managers:
        worker_cp_manager.purge_old_checkpoints(keep_num)

  def __reset_all_checkpoint_status(self):
    self._pending_ready_offsets.clear()
    self._pending_checkpoints.clear()
    self._checkpoint_states = [CheckpointState.WAITING for _ in range(self._worker_num)]

  def __check_all_waiting(self):
    for state in self._checkpoint_states:
      if state is not CheckpointState.WAITING:
        return False
    return True
