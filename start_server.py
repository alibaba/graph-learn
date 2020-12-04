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

import os
import base64
import json
import sys
import graphlearn as gl
import time


def launch_server():
  handle_str = sys.argv[1]
  train_config_str = sys.argv[2]
  s = base64.b64decode(handle_str).decode('utf-8')
  handle = json.loads(s)
  s = base64.b64decode(train_config_str).decode('utf-8')
  train_config = json.loads(s)
  pod_index = int(sys.argv[3])
  handle['server'] = handle['hosts']
  g = gl.Graph().vineyard(handle)
  for node_spec in train_config['node_spces']:
    g = g.node_attributes(
      node_spec['node_type'],
      features=node_spec['features'],
      n_int=node_spec.get('n_int', 0),
      n_float=node_spec.get('n_float', 0),
      n_string=node_spec.get('n_string', 0))

  for node_view in train_config['node_views']:
    g = g.node_view(
      node_view_type = node_view['view_type'],
      node_type = node_view['node_type'],
      seed = node_view.get('seed', 0),
      nsplit = node_view['nsplit'],
      split_range = node_view['split_range'])

  g = g.init_vineyard(server_index=pod_index, worker_count=handle['client_count'])
  print('servers', handle['server'])
  time.sleep(100000)

if __name__ == "__main__":
  launch_server()
