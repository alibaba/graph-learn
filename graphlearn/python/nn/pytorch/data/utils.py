# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
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

import os
import socket
import warnings
import multiprocessing as mp

try:
  import torch
except ImportError:
  pass
import torch.distributed as dist


SERVER_LAUNCHED = False

CLUSTER_SPEC = None
WORLD_SIZE = None
RANK = None
NUM_CLIENT = None
STATS_DICT = []


def get_world_size():
  global WORLD_SIZE
  if WORLD_SIZE is None:
    if dist.is_initialized():
      WORLD_SIZE = dist.get_world_size()
    else:
      WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
  return WORLD_SIZE

def get_rank():
  global RANK
  if RANK is None:
    if dist.is_initialized():
      RANK = dist.get_rank()
    else:
      RANK = int(os.getenv('RANK', 0))
  return RANK

def get_num_client():
  global NUM_CLIENT
  if NUM_CLIENT is None:
    NUM_CLIENT = int(os.getenv('GL_NUM_CLIENT', 1))
  return NUM_CLIENT

def get_cluster_spec():
  global CLUSTER_SPEC
  if CLUSTER_SPEC is None:
    world_size = get_world_size()
    rank = get_rank()
    num_client = get_num_client()
    gl_server_info = bootstrap(world_size, rank)
    CLUSTER_SPEC = {"server": gl_server_info, 'client_count': world_size * num_client}
  return CLUSTER_SPEC

def set_client_num(n):
  assert isinstance(n, int), 'client_num should be int, not {}'.format(str(type(n)))
  global NUM_CLIENT
  if NUM_CLIENT is not None:
    warnings.warn('graph learn client number has been configured')
  else:
    NUM_CLIENT = n

def bootstrap(world_size, rank):
  def get_free_port(host='127.0.0.1'):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

  def addr_to_tensor(ip, port):
    addr_array = [int(i) for i in (ip.split('.'))] + [int(port)]
    addr_tensor = torch.tensor(addr_array, dtype=torch.int)
    return addr_tensor

  def tensor_to_addr(tensor):
    addr_array = tensor.tolist()
    addr = '.'.join([str(i) for i in addr_array[:-1]]) + ':' + str(addr_array[-1])
    return addr

  def exchange_gl_server_info(addr_tensor ,world_size, rank):
    comm_tensor = torch.zeros([world_size, 5], dtype=torch.int32)
    comm_tensor[rank] = addr_tensor
    if dist.get_backend() == dist.Backend.NCCL:
      comm_tensor = comm_tensor.cuda()
    dist.all_reduce(comm_tensor, op=dist.ReduceOp.MAX)
    cluster_server_info = ','.join([tensor_to_addr(t) for t in comm_tensor])
    return cluster_server_info

  local_ip = socket.gethostbyname(socket.gethostname())
  port = str(get_free_port(local_ip))
  if not dist.is_initialized(): # stand-alone
    return local_ip + ':' + port
  gl_server_info = exchange_gl_server_info(addr_to_tensor(local_ip, port), world_size, rank)
  return gl_server_info

def _server_manager(graph, cluster, task_index, counts_dict):
  graph.init(cluster=cluster, job_name="server", task_index=task_index)
  counts_dict.update(graph.server_get_stats())
  graph.close()

def launch_server(g, cluster=None, task_index=None):
  global SERVER_LAUNCHED
  global STATS_DICT
  if SERVER_LAUNCHED:
    raise RuntimeError('duplicate server launch detected')
  if cluster is None:
    cluster = get_cluster_spec()
    task_index = get_rank()
  elif task_index is None:
    raise UserWarning('task_index should be explicitly defined when cluster defined by user')
  
  counts_dict = mp.Manager().dict()
  p = mp.Process(target=_server_manager, args=(g, cluster, task_index, counts_dict))
  p.daemon = True
  p.start()
  SERVER_LAUNCHED = True
  STATS_DICT.append(counts_dict)

def get_counts():
  return STATS_DICT[0]

def is_server_launched():
  global SERVER_LAUNCHED
  return SERVER_LAUNCHED
