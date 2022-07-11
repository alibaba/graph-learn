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

import json
import logging
import os
import sys
import time
from threading import Thread
from urllib import request

import grpc
import coordinator_pb2
import coordinator_pb2_grpc

name_map = {
  coordinator_pb2.DataLoader : "DataLoader",
  coordinator_pb2.Sampling : "Sampling",
  coordinator_pb2.Serving : "Serving"
}

def get_init_info(stub, worker_type, worker_ip, pod_id):
  request = coordinator_pb2.GetInitInfoRequestPb(
    worker_type=worker_type, worker_ip=worker_ip, pod_id=pod_id)
  ret = stub.GetInitInfo(request)

def report_inited(stub, worker_id, worker_type):
  request = coordinator_pb2.ReportInitedRequestPb(
    worker_id=worker_id, worker_type=worker_type)
  ret = stub.ReportInited(request)
  print("report_inited for", name_map[worker_type], "log in  client_output.{}-{}"
        .format(name_map[worker_type], worker_id))
  with open("client_output.{}-{}".format(name_map[worker_type], worker_id), 'a') as f:
    f.write("\n------Reach Global Inited Barrier.------\n")
    f.write("terminate_service: {}\n".format(str(ret.terminate_service)))

def report_sever_is_ready(stub, worker_id, worker_type):
  request = coordinator_pb2.ServerIsReadyRequestPb(
    worker_id=worker_id, worker_type=worker_type)
  ret = stub.ReportServerIsReady(request)
  print("report_sever_is_ready for", name_map[worker_type], "log in client_output.{}-{}"
        .format(name_map[worker_type], worker_id))
  with open("client_output.{}-{}".format(name_map[worker_type], worker_id), 'a') as f:
    f.write("\n------Reach Global Ready Barrier.------\n")
    f.write("terminate_service: {}\n".format(str(ret.terminate_service)))

def get_check_ready_info(stub, worker_id, worker_type):
  request = coordinator_pb2.CheckReadyInfoRequestPb(
    worker_id=worker_id, worker_type=worker_type)
  ret = stub.GetCheckReadyInfo(request)
  print("get_check_ready_info for", name_map[worker_type], "log in client_output.{}-{}"
        .format(name_map[worker_type], worker_id))
  with open("client_output.{}-{}".format(name_map[worker_type], worker_id), 'a') as f:
    f.write("\n------------------Get Check Ready Info.------------\n".format(worker_type))
    f.write("terminate_service: {}\n".format(str(ret.terminate_service)))

def report_statistics(stub, worker_id, worker_type):
  request = coordinator_pb2.ReportStatisticsRequestPb(worker_id=worker_id, worker_type=worker_type)
  ret = stub.ReportStatistics(request)
  print("report_statistics for", name_map[worker_type], "log in client_output.{}-{}"
        .format(name_map[worker_type], worker_id))
  with open("client_output.{}-{}".format(name_map[worker_type], worker_id), 'a') as f:
    f.write("\nReport Statistics with : \n"
            "worker_id: {}, \n"
            "worker_type: {}. \n"
            "Get Return termination: {}\n".format(
              worker_id, worker_type, str(ret.terminate_service)))

def datalaoder(worker_id):
  with grpc.insecure_channel('localhost:50051') as channel:
    stub = coordinator_pb2_grpc.CoordinatorStub(channel)
    get_init_info(stub, coordinator_pb2.DataLoader, "", worker_id)
    report_sever_is_ready(stub, worker_id, coordinator_pb2.DataLoader)
    report_statistics(stub, worker_id, coordinator_pb2.DataLoader)
    time.sleep(4)
    report_statistics(stub, worker_id, coordinator_pb2.DataLoader)


def sampling(worker_id): # 3
  with grpc.insecure_channel('localhost:50051') as channel:
    stub = coordinator_pb2_grpc.CoordinatorStub(channel)
    get_init_info(stub, coordinator_pb2.Sampling, "127.0.0.1", worker_id)
    report_inited(stub, worker_id, coordinator_pb2.Sampling)
    get_check_ready_info(stub, worker_id, coordinator_pb2.Sampling)
    report_sever_is_ready(stub, worker_id, coordinator_pb2.Sampling)
    report_statistics(stub, worker_id, coordinator_pb2.Sampling)
    time.sleep(4)
    report_statistics(stub, worker_id, coordinator_pb2.Sampling)


def serving(worker_id): # 2
  with grpc.insecure_channel('localhost:50051') as channel:
    stub = coordinator_pb2_grpc.CoordinatorStub(channel)
    get_init_info(stub, coordinator_pb2.Serving, "127.0.0.1", worker_id)
    report_inited(stub, worker_id, coordinator_pb2.Serving)
    get_check_ready_info(stub, worker_id, coordinator_pb2.Serving)
    report_sever_is_ready(stub, worker_id, coordinator_pb2.Serving)
    report_statistics(stub, worker_id, coordinator_pb2.Serving)
    time.sleep(4)
    report_statistics(stub, worker_id, coordinator_pb2.Serving)

def main():
  threads = []
  threads.append(Thread(target=datalaoder, args=(0, )))
  threads.append(Thread(target=sampling, args=(0,)))
  threads.append(Thread(target=sampling, args=(1,)))
  threads.append(Thread(target=serving, args=(0,)))
  threads.append(Thread(target=serving, args=(1,)))
  threads.append(Thread(target=serving, args=(2,)))
  threads.append(Thread(target=serving, args=(3,)))
  for t in threads:
    t.start()
  cur_path = sys.path[0]
  json_file = os.path.join(cur_path, "../../../conf/install_query.e2e.json")
  install_query_req = None
  with open (json_file, 'r') as f:
    install_query_req = json.loads(f.read())
  # TODO(@Seventeen17): fixme
  os.system("curl -X POST -H \"Content-Type: text/plain\" -d \'"
            + json.dumps(install_query_req) + "\' http://127.0.0.1:8080/administration/init/")
  time.sleep(2)
  os.system("curl -X POST -H \"Content-Type: text/plain\" -d \'\' http://127.0.0.1:8080/administration/terminate/")
  for t in threads:
    t.join()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    os.system("rm -rf client_output.*")
    main()
