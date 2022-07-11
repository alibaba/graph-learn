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

import grpc
import coordinator_pb2
import coordinator_pb2_grpc

from checkpoint import SubStateCheckpointManager

from concurrent import futures
import logging
import os
import threading
import time
from enum import IntEnum


class WorkerState(IntEnum):
  TERMINATED = 0
  REGISTERED = 1
  INITED = 2
  READY = 3


class SubServiceState(object):
  def __init__(self, num_workers):
    self.name = ""
    self.upstream_worker_type = None
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

  def check_is_inited(self, worker_id):
    with self._lock:
      return int(self._states[worker_id]) == int(WorkerState.INITED)

  def check_is_ready(self, worker_id):
    with self._lock:
      return int(self._states[worker_id]) == int(WorkerState.READY)

  def check_all_have_registered(self):
    with self._lock:
      return self._check_states_of_all(WorkerState.REGISTERED)

  def check_all_have_inited(self):
    with self._lock:
      return self._check_states_of_all(WorkerState.INITED)

  def check_all_have_ready(self):
    with self._lock:
      return self._check_states_of_all(WorkerState.READY)

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

  def set_inited(self, worker_id):
    with self._lock:
      self._states[worker_id] = WorkerState.INITED
      logging.info("{}-{} is inited".format(self.name, worker_id))

  def set_ready(self, worker_id):
    with self._lock:
      self._states[worker_id] = WorkerState.READY
      logging.info("{}-{} is ready".format(self.name, worker_id))

  def _check_states_of_all(self, expected_state):
    for s in self._states:
      if int(s) < int(expected_state):
        return False
    return True


class DataLoaderState(SubServiceState):
  def __init__(self, configs):
    num_workers = configs.get("worker_num")
    super().__init__(num_workers)
    self.name = "DataLoader"
    self.upstream_worker_type = None
    self.checkpoint_manager = None
    self.info = configs
    logging.info("Configure service with {} DataLoader Workers.".format(num_workers))


class SamplingWorkerState(SubServiceState):
  def __init__(self, configs, meta_root_dir):
    num_workers = configs.get("worker_num")
    super().__init__(num_workers)
    self.name = "SamplingWorker"
    self.upstream_worker_type = coordinator_pb2.DataLoader
    self.checkpoint_manager = SubStateCheckpointManager(
      worker_num=num_workers,
      meta_dir=os.path.join(meta_root_dir, self.name),
      kafka_pids_group=configs.get("upstream").get("sub_kafka_pids"),
      store_pids_group=configs.get("store_partition").get("managed_pids_group"),
      independent=False
    )
    self.info = configs
    logging.info("Configure service with {} Sampling Workers.".format(num_workers))

  def get_composed_ip_list(self):
    composed_ip_list = []
    for i in range(self.num_workers):
      composed_ip_list.append("{} {} {}".format(i, self.info.get("num_local_shards"), self._ip_addrs[i]))
    return composed_ip_list


class ServingWorkerState(SubServiceState):
  def __init__(self, configs, meta_root_dir):
    num_workers = configs.get("worker_num")
    super().__init__(num_workers)
    self.name = "ServingWorker"
    self.upstream_worker_type = coordinator_pb2.Sampling
    self.checkpoint_manager = SubStateCheckpointManager(
      worker_num=num_workers,
      meta_dir=os.path.join(meta_root_dir, self.name),
      kafka_pids_group=configs.get("upstream").get("sub_kafka_pids"),
      store_pids_group=configs.get("store_partition").get("managed_pids_group"),
      independent=True
    )
    self.info = configs
    logging.info("Configure service with {} Serving Workers.".format(num_workers))


def set_upstream_init_info(pb, info, worker_id):
  pb.sub_kafka_servers.extend(info.get("sub_kafka_servers"))
  pb.sub_kafka_topic = info.get("sub_kafka_topic")
  pb.sub_kafka_partition_num = info.get("sub_kafka_partition_num")
  pb.sub_kafka_pids.extend(info.get("sub_kafka_pids")[worker_id])


def set_downstream_init_info(pb, info):
  pb.store_partition_strategy = info.get("store_partition_strategy")
  pb.store_partition_num = info.get("store_partition_num")
  pb.worker_partition_strategy = info.get("worker_partition_strategy")
  pb.worker_partition_num = info.get("worker_partition_num")
  pb.pub_kafka_servers.extend(info.get("pub_kafka_servers"))
  pb.pub_kafka_topic = info.get("pub_kafka_topic")
  pb.pub_kafka_partition_num = info.get("pub_kafka_partition_num")
  pb.pub_kafka_pids.extend(info.get("pub_kafka_pids"))


def set_store_partition_info(pb, info):
  pb.partition_strategy = info.get("partition_strategy")
  pb.partition_num = info.get("partition_num")
  for k, v in info.get("managed_pids_group").items():
    entry = pb.managed_pids_group.add()
    entry.worker_id = k
    entry.pids.extend(v)


class CoordinatorServicer(coordinator_pb2_grpc.CoordinatorServicer):
  def __init__(self, configs):
    # States for sub-services.
    meta_root_dir = configs.get("meta_dir")
    self._sub_states = {
      coordinator_pb2.DataLoader: DataLoaderState(configs.get("data_loading")),
      coordinator_pb2.Sampling: SamplingWorkerState(configs.get("sampling"), meta_root_dir),
      coordinator_pb2.Serving: ServingWorkerState(configs.get("serving"), meta_root_dir)
    }
    self._sub_names = {
      coordinator_pb2.DataLoader: "DataLoader",
      coordinator_pb2.Sampling: "SamplingWorker",
      coordinator_pb2.Serving: "ServingWorker"
    }

    self._query_plan = None
    self._install_query_sem = threading.Semaphore()
    self._install_query_sem.acquire()

    # \ready_sem semaphore is for HTTP service.
    # When client send a init request, the response will not return until
    # \ready_sem is released.
    # Meanwhile, \ready_sem will be released when all the workers are ready.
    self.ready_sem = threading.Semaphore()
    self.ready_sem.acquire()

  def init_query(self, query_plan):
    logging.info("Initialize query plan: {}\n".format(query_plan))
    self._query_plan = query_plan
    self._install_query_sem.release()

  def start_checkpointing(self):
    logging.info("Start to create new checkpoint ...")
    sampling_res = self._sub_states.get(coordinator_pb2.Sampling).checkpoint_manager.start_checkpointing()
    serving_res = self._sub_states.get(coordinator_pb2.Serving).checkpoint_manager.start_checkpointing()
    return sampling_res, serving_res

  def purge_old_checkpoints(self, keep_num):
    logging.info("Purging old checkpoints by reserving latest {} versions ...".format(keep_num))
    self._sub_states.get(coordinator_pb2.Sampling).checkpoint_manager.purge_old_checkpoints(keep_num)
    self._sub_states.get(coordinator_pb2.Serving).checkpoint_manager.purge_old_checkpoints(keep_num)

  def RegisterWorker(self, request, context):
    worker_type = request.worker_type
    worker_id = request.worker_id
    sub_state = self._sub_states.get(worker_type)
    logging.info("Receiving RegisterWorker request from {}-{}.".format(sub_state.name, worker_id))

    successful = sub_state.register_worker(worker_id, request.worker_ip, not (worker_type == coordinator_pb2.Sampling))
    register_info_pb = coordinator_pb2.RegisterWorkerResponsePb(
      suc=successful, num_workers=sub_state.num_workers)
    return register_info_pb

  def GetInitInfo(self, request, context):
    worker_type = request.worker_type
    worker_id = request.worker_id
    sub_state = self._sub_states.get(worker_type)
    logging.info("Receiving GetInitInfo Request from {}-{}.".format(sub_state.name, worker_id))

    init_info_pb = coordinator_pb2.GetInitInfoResponsePb()
    if worker_type == coordinator_pb2.DataLoader:
      data_loading_info_pb = init_info_pb.dataloader_info
      set_downstream_init_info(data_loading_info_pb.downstream_info, sub_state.info.get("downstream"))
    elif worker_type == coordinator_pb2.Sampling:
      sampling_info_pb = init_info_pb.sampling_info
      # Set non-blocking infos
      sampling_info_pb.num_local_shards = sub_state.info.get("num_local_shards")
      set_store_partition_info(sampling_info_pb.store_partition_info, sub_state.info.get("store_partition"))
      set_upstream_init_info(sampling_info_pb.upstream_info, sub_state.info.get("upstream"), worker_id)
      set_downstream_init_info(sampling_info_pb.downstream_info, sub_state.info.get("downstream"))
      sub_state.checkpoint_manager.set_init_pb_with_latest_checkpoint(worker_id, sampling_info_pb.checkpoint_info)
      # Wait until all sampling workers are registered and set ip address list.
      while sub_state.check_is_registered(worker_id):
        if sub_state.check_all_have_registered():
          sampling_info_pb.ipaddrs.extend(sub_state.get_composed_ip_list())
          break
        time.sleep(1)
      # Wait until query plan is installed.
      while sub_state.check_is_registered(worker_id):
        if self._install_query_sem.acquire(blocking=False):
          self._install_query_sem.release()
          sampling_info_pb.query_plan = self._query_plan
          break
        time.sleep(1)
    elif worker_type == coordinator_pb2.Serving:
      serving_info_pb = init_info_pb.serving_info
      # Set non-blocking infos
      serving_info_pb.num_local_shards = sub_state.info.get("num_local_shards")
      set_store_partition_info(serving_info_pb.store_partition_info, sub_state.info.get("store_partition"))
      set_upstream_init_info(serving_info_pb.upstream_info, sub_state.info.get("upstream"), worker_id)
      sub_state.checkpoint_manager.set_init_pb_with_latest_checkpoint(worker_id, serving_info_pb.checkpoint_info)
      # Wait until query plan is installed.
      while sub_state.check_is_registered(worker_id):
        if self._install_query_sem.acquire(blocking=False):
          self._install_query_sem.release()
          serving_info_pb.query_plan = self._query_plan
          break
        time.sleep(1)
    # Set terminate state
    init_info_pb.terminate_service = sub_state.check_is_terminated(worker_id)
    if init_info_pb.terminate_service:
      logging.info("Notify {}-{} to restart.".format(sub_state.name, worker_id))
    return init_info_pb

  def ReportInited(self, request, context):
    worker_id = request.worker_id
    worker_type = request.worker_type
    sub_state = self._sub_states.get(worker_type)
    logging.info("Receiving ReportInited Request from {}-{}.".format(sub_state.name, worker_id))

    if sub_state.check_is_registered(worker_id):
      sub_state.set_inited(worker_id)
      # sampling workers need to wait for all to be initialized。
      if worker_type == coordinator_pb2.Sampling:
        while sub_state.check_is_inited(worker_id):
          if sub_state.check_all_have_inited():
            break
          time.sleep(1)
    need_terminate = sub_state.check_is_terminated(worker_id)
    if need_terminate:
      logging.info("Notify {}-{} to restart.".format(sub_state.name, worker_id))
    return coordinator_pb2.ReportInitedResponsePb(terminate_service=need_terminate)

  def GetCheckReadyInfo(self, request, context):
    worker_type = request.worker_type
    worker_id = request.worker_id
    sub_state = self._sub_states.get(worker_type)
    logging.info("Receiving GetCheckReadyInfo Request from {}-{}.".format(sub_state.name, worker_id))

    upstream_worker_type = sub_state.upstream_worker_type
    if upstream_worker_type is not None:
      upstream_sub_state = self._sub_states.get(upstream_worker_type)
      while sub_state.check_is_inited(worker_id):
        if upstream_sub_state.check_all_have_ready():
          break
        time.sleep(1)
    need_terminate = sub_state.check_is_terminated(worker_id)
    if need_terminate:
      logging.info("Notify {}-{} to restart.".format(sub_state.name, worker_id))
    return coordinator_pb2.CheckReadyInfoResponsePb(terminate_service=need_terminate)

  def ReportServerIsReady(self, request, context):
    worker_type = request.worker_type
    worker_id = request.worker_id
    sub_state = self._sub_states.get(worker_type)
    logging.info("Receiving ReportServerIsReady Request from {}-{}.".format(sub_state.name, worker_id))

    if sub_state.check_is_inited(worker_id):
      sub_state.set_ready(request.worker_id)
      # Notify coordinator that the service is ready.
      if request.worker_type == coordinator_pb2.Serving:
        self.ready_sem.release()
    need_terminate = sub_state.check_is_terminated(worker_id)
    if need_terminate:
      logging.info("Notify {}-{} to restart.".format(sub_state.name, worker_id))
    return coordinator_pb2.ServerIsReadyResponsePb(terminate_service=need_terminate)

  def ReportStatistics(self, request, context):
    worker_type = request.worker_type
    worker_id = request.worker_id
    sub_state = self._sub_states.get(worker_type)
    checkpoint_manager = sub_state.checkpoint_manager
    if checkpoint_manager is not None:
      checkpoint_manager.update_kafka_ready_offsets(worker_id, request.ready_kafka_offsets)
      need_backup = checkpoint_manager.check_for_backup(worker_id)
    else:
      need_backup = False
    if need_backup:
      logging.info("Notify {}-{} to backup.".format(sub_state.name, worker_id))
    need_terminate = sub_state.check_is_terminated(worker_id)
    if need_terminate:
      logging.info("Notify {}-{} to restart.".format(sub_state.name, worker_id))
    return coordinator_pb2.ReportStatisticsResponsePb(
      terminate_service=need_terminate, do_backup=need_backup)

  def ReportBackupFinished(self, request, context):
    worker_type = request.worker_type
    worker_id = request.worker_id
    sub_state = self._sub_states.get(worker_type)
    logging.info("Receiving ReportBackupFinished Request from {}-{}.".format(sub_state.name, worker_id))

    valid = sub_state.checkpoint_manager.report_worker_backup_with_pb(
      worker_id, request.sample_store_backups, request.subs_table_backups)
    return coordinator_pb2.ReportBackupFinishedResponsePb(is_valid=valid)


class CoordinatorGrpcService(object):
  def __init__(self, configs, port=50051, max_workers=10):
    self._port = port
    self._server = grpc.server(
      futures.ThreadPoolExecutor(max_workers=max_workers),
      options=(
        # send keepalive ping every 60 second, default is 2 hours
        ('grpc.keepalive_time_ms', 60000),
        # keepalive ping time out after 5 seconds, default is 20 seconds
        ('grpc.keepalive_timeout_ms', 5000),
        # allow keepalive pings when there's no gRPC calls
        ('grpc.keepalive_permit_without_calls', True)
      )
    )
    self._grpc_servicer = CoordinatorServicer(configs)
    coordinator_pb2_grpc.add_CoordinatorServicer_to_server(self._grpc_servicer, self._server)
    self._server.add_insecure_port('[::]:{}'.format(self._port))

  def start(self):
    logging.info("Grpc Server for Coordinator running on port {}.\n".format(self._port))
    self._server.start()
    self._server.wait_for_termination()

  def wait_for_ready(self):
    logging.info("Waiting for service ready....")
    self._grpc_servicer.ready_sem.acquire()
    self._grpc_servicer.ready_sem.release()
    logging.info("Service is ready for serving :)")

  def stop(self):
    self._server.stop(0)
    logging.info('Stopping Grpc Server...\n')

  def init_query(self, query_plan):
    self._grpc_servicer.init_query(query_plan)

  def start_checkpointing(self):
    return self._grpc_servicer.start_checkpointing()

  def purge_old_checkpoints(self, keep_num):
    self._grpc_servicer.purge_old_checkpoints(keep_num)
