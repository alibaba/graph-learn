# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
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

import base64
import graphlearn as gl
import json
import numpy as np

from graphlearn.python.config import *
from graphlearn.python.graph import Graph
from graphlearn.python.decoder import Decoder

def init_graph_from_handle(handle, server_index):
  """ Used in server.
      Parse the handle passed from vineyard and return gl.Graph object.
  """
  if isinstance(handle, dict):
    obj = handle
  else:
    s = base64.b64decode(handle).decode('utf-8')
    obj = json.loads(s)

  gl.set_storage_mode(8)
  gl.set_vineyard_graph_id(obj['vineyard_id'])
  gl.set_vineyard_ipc_socket(obj['vineyard_socket'])

  g = Graph()
  gl.set_tracker_mode(0)
  # here client_count is a placeholder without actual meaning
  cluster = {'server': obj['server'], 'client': obj['client']}
  g.init(cluster=cluster, task_index=server_index, job_name="server")
  return g

def get_graph_from_handle(handle, worker_index, worker_count, standalone=False):
  """ Used in client.
      Parse the handle passed from vineyard and return gl.Graph object.

      Paramters
      ---------
      standalone: single machine mode
  """
  if isinstance(handle, dict):
    obj = handle
  else:
    s = base64.b64decode(handle).decode('utf-8')
    obj = json.loads(s)

  gl.set_storage_mode(8)
  gl.set_vineyard_graph_id(obj['vineyard_id'])
  gl.set_vineyard_ipc_socket(obj['vineyard_socket'])

  g = Graph()
  for node_info in obj['node_schema']:
    confs = node_info.split(':')
    if len(confs) != 6:
      continue
    else:
      node_type = confs[0]
      weighted = confs[1] == 'true'
      labeled = confs[2] == 'true'
      n_int = int(confs[3])
      n_float = int(confs[4])
      n_string = int(confs[5])
      g.node(source='', node_type=node_type,
             decoder=get_decoder(weighted, labeled, n_int, n_float, n_string))

  for edge_info in obj['edge_schema']:
    confs = edge_info.split(':')
    if len(confs) != 8:
      continue
    else:
      src_node_type = confs[0]
      edge_type = confs[1]
      dst_node_type = confs[2]
      weighted = confs[3] == 'true'
      labeled = confs[4] == 'true'
      n_int = int(confs[5])
      n_float = int(confs[6])
      n_string = int(confs[7])
      g.edge(source='', edge_type=(src_node_type, dst_node_type, edge_type),
             decoder=get_decoder(weighted, labeled, n_int, n_float, n_string))

  gl.set_tracker_mode(0)
  if standalone:
    g.init()
  else:
    cluster = {'server': obj['server'], 'client': obj['client']}
    g.init(cluster=cluster, task_index=worker_index, job_name="client")
  return g

def get_decoder(weighted, labeled, n_int, n_float, n_string):
  attr_types = []
  if n_int == 0 and n_float == 0 and n_string == 0:
    attr_types = None
  else:
    attr_types.extend(["int"] * n_int)
    attr_types.extend(["float"] * n_float)
    attr_types.extend(["string"] * n_string)

  return Decoder(weighted, labeled, attr_types)
