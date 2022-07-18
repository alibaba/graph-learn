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

from concurrent import futures
import logging
import threading
import time


def set_upstream_init_info(pb, info, worker_id):
  pb.sub_kafka_servers.extend(info.get("sub_kafka_servers"))
  pb.sub_kafka_topic = info.get("sub_kafka_topic")
  pb.sub_kafka_partition_num = info.get("sub_kafka_partition_num")
  pb.sub_kafka_pids.extend(info.get("sub_kafka_pids")[worker_id])


def set_downstream_kafka_info(pb, info):
  pb.pub_kafka_servers.extend(info.get("pub_kafka_servers"))
  pb.pub_kafka_topic = info.get("pub_kafka_topic")
  pb.pub_kafka_partition_num = info.get("pub_kafka_partition_num")


def set_downstream_partition_info(pb, info):
  pb.worker_partition_strategy = info.get("worker_partition_strategy")
  pb.worker_partition_num = info.get("worker_partition_num")
  pb.kafka_to_wid.extend(info.get("kafka_to_wid"))


def set_store_partition_info(pb, info):
  pb.partition_strategy = info.get("partition_strategy")
  pb.partition_num = info.get("partition_num")
  for k, v in info.get("managed_pids_group").items():
    entry = pb.managed_pids_group.add()
    entry.worker_id = k
    entry.pids.extend(v)


class CoordinatorServicer(coordinator_pb2_grpc.CoordinatorServicer):
  def __init__(self, sampling_manager, serving_manager):
    # Managers for sub-services.
    self._sub_managers = {
      coordinator_pb2.Sampling: sampling_manager,
      coordinator_pb2.Serving: serving_manager
    }
    self._query_plan = None
    self._install_query_sem = threading.Semaphore()
    self._install_query_sem.acquire()

  def init_query(self, query_plan):
    logging.info("Initialize query plan: {}\n".format(query_plan))
    self._query_plan = query_plan
    self._install_query_sem.release()

  def start_checkpointing(self):
    logging.info("Start to create new checkpoint ...")
    sampling_res = self._sub_managers.get(coordinator_pb2.Sampling).checkpoint_manager.start_checkpointing()
    serving_res = self._sub_managers.get(coordinator_pb2.Serving).checkpoint_manager.start_checkpointing()
    return sampling_res, serving_res

  def purge_old_checkpoints(self, keep_num):
    logging.info("Purging old checkpoints by reserving latest {} versions ...".format(keep_num))
    self._sub_managers.get(coordinator_pb2.Sampling).checkpoint_manager.purge_old_checkpoints(keep_num)
    self._sub_managers.get(coordinator_pb2.Serving).checkpoint_manager.purge_old_checkpoints(keep_num)

  def RegisterWorker(self, request, context):
    worker_type = request.worker_type
    worker_id = request.worker_id
    sub_mgr = self._sub_managers.get(worker_type)
    logging.info("Receiving RegisterWorker request from {}-{}.".format(sub_mgr.name, worker_id))

    successful = sub_mgr.register_worker(worker_id, request.worker_ip, not (worker_type == coordinator_pb2.Sampling))
    register_info_pb = coordinator_pb2.RegisterWorkerResponsePb(
      suc=successful, num_workers=sub_mgr.num_workers)
    return register_info_pb

  def GetInitInfo(self, request, context):
    worker_type = request.worker_type
    worker_id = request.worker_id
    sub_mgr = self._sub_managers.get(worker_type)
    logging.info("Receiving GetInitInfo Request from {}-{}.".format(sub_mgr.name, worker_id))

    init_info_pb = coordinator_pb2.GetInitInfoResponsePb()
    if worker_type == coordinator_pb2.Sampling:
      sampling_info_pb = init_info_pb.sampling_info
      # Set non-blocking infos
      sampling_info_pb.num_local_shards = sub_mgr.info.get("num_local_shards")
      set_store_partition_info(sampling_info_pb.store_partition_info, sub_mgr.info.get("store_partition"))
      set_upstream_init_info(sampling_info_pb.upstream_info, sub_mgr.info.get("upstream"), worker_id)
      set_downstream_kafka_info(
        sampling_info_pb.ds_kafka_info, sub_mgr.info.get("downstream").get("kafka"))
      set_downstream_partition_info(
        sampling_info_pb.ds_partition_info, sub_mgr.info.get("downstream").get("partition"))
      sub_mgr.checkpoint_manager.set_init_pb_with_latest_checkpoint(worker_id, sampling_info_pb.checkpoint_info)
      # Wait until all sampling workers are registered and set ip address list.
      while sub_mgr.check_is_registered(worker_id):
        if sub_mgr.check_all_have_registered():
          sampling_info_pb.ipaddrs.extend(sub_mgr.get_composed_ip_list())
          break
        time.sleep(1)
      # Wait until query plan is installed.
      while sub_mgr.check_is_registered(worker_id):
        if self._install_query_sem.acquire(blocking=False):
          self._install_query_sem.release()
          sampling_info_pb.query_plan = self._query_plan
          break
        time.sleep(1)
    elif worker_type == coordinator_pb2.Serving:
      serving_info_pb = init_info_pb.serving_info
      # Set non-blocking infos
      serving_info_pb.num_local_shards = sub_mgr.info.get("num_local_shards")
      set_store_partition_info(serving_info_pb.store_partition_info, sub_mgr.info.get("store_partition"))
      set_upstream_init_info(serving_info_pb.upstream_info, sub_mgr.info.get("upstream"), worker_id)
      sub_mgr.checkpoint_manager.set_init_pb_with_latest_checkpoint(worker_id, serving_info_pb.checkpoint_info)
      # Wait until query plan is installed.
      while sub_mgr.check_is_registered(worker_id):
        if self._install_query_sem.acquire(blocking=False):
          self._install_query_sem.release()
          serving_info_pb.query_plan = self._query_plan
          break
        time.sleep(1)
    # Set terminate state
    init_info_pb.terminate_service = sub_mgr.check_is_terminated(worker_id)
    if init_info_pb.terminate_service:
      logging.info("Notify {}-{} to restart.".format(sub_mgr.name, worker_id))
    return init_info_pb

  def ReportStarted(self, request, context):
    worker_id = request.worker_id
    worker_type = request.worker_type
    sub_mgr = self._sub_managers.get(worker_type)
    logging.info("Receiving ReportStarted Request from {}-{}.".format(sub_mgr.name, worker_id))

    if sub_mgr.check_is_registered(worker_id):
      sub_mgr.set_started(worker_id)
    need_terminate = sub_mgr.check_is_terminated(worker_id)
    if need_terminate:
      logging.info("Notify {}-{} to restart.".format(sub_mgr.name, worker_id))
    return coordinator_pb2.ReportStartedResponsePb(terminate_service=need_terminate)

  def ReportStatistics(self, request, context):
    worker_type = request.worker_type
    worker_id = request.worker_id
    sub_mgr = self._sub_managers.get(worker_type)

    sub_mgr.checkpoint_manager.update_kafka_ready_offsets(worker_id, request.ready_kafka_offsets)
    need_backup = sub_mgr.checkpoint_manager.check_for_backup(worker_id)
    if need_backup:
      logging.info("Notify {}-{} to backup.".format(sub_mgr.name, worker_id))
    need_terminate = sub_mgr.check_is_terminated(worker_id)
    if need_terminate:
      logging.info("Notify {}-{} to restart.".format(sub_mgr.name, worker_id))
    return coordinator_pb2.ReportStatisticsResponsePb(
      terminate_service=need_terminate, do_backup=need_backup)

  def ReportBackupFinished(self, request, context):
    worker_type = request.worker_type
    worker_id = request.worker_id
    sub_mgr = self._sub_managers.get(worker_type)
    logging.info("Receiving ReportBackupFinished Request from {}-{}.".format(sub_mgr.name, worker_id))

    valid = sub_mgr.checkpoint_manager.report_worker_backup_with_pb(
      worker_id, request.sample_store_backups, request.subs_table_backups)
    return coordinator_pb2.ReportBackupFinishedResponsePb(is_valid=valid)


class CoordinatorGrpcService(object):
  def __init__(self, port, sampling_manager, serving_manager, max_workers=10):
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
    self._grpc_servicer = CoordinatorServicer(sampling_manager, serving_manager)
    coordinator_pb2_grpc.add_CoordinatorServicer_to_server(self._grpc_servicer, self._server)
    self._server.add_insecure_port('[::]:{}'.format(self._port))

  def start(self):
    logging.info("Grpc Server for Coordinator running on port {}.\n".format(self._port))
    self._server.start()
    self._server.wait_for_termination()

  def stop(self):
    self._server.stop(0)
    logging.info('Stopping Grpc Server...\n')

  def init_query(self, query_plan):
    self._grpc_servicer.init_query(query_plan)

  def start_checkpointing(self):
    return self._grpc_servicer.start_checkpointing()

  def purge_old_checkpoints(self, keep_num):
    self._grpc_servicer.purge_old_checkpoints(keep_num)
