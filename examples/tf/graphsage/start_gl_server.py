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
  s = base64.b64decode(handle_str).decode('utf-8')
  handle = json.loads(s)
  handle['pod_index'] = int(sys.argv[2])
  features = []
  for i in range(128):
    features.append("feat_" + str(i))
  features.append("KC")
  features.append("TC")
  hosts = handle['hosts'].split(',')
  handle['server'] = ','.join(["{}:{}".format(pod_name, 8000 + index) for index, pod_name in enumerate(hosts[0:])])
  g = gl.Graph().vineyard(handle, nodes=["paper"], edges=["cites"]) \
      .node_attributes("paper", features, n_int=2, n_float=128, n_string=0) \
      .init_vineyard(server_index=handle['pod_index'], worker_count=handle['client_count'])
  print('servers', handle['server'])
  time.sleep(100000)

if __name__ == "__main__":
  launch_server()
