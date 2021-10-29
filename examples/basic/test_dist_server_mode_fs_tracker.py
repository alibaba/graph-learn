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

import getopt
import os
import sys

import graphlearn as gl
from query_examples import *


def main(argv):
  cur_path = sys.path[0]

  server_count = -1
  client_count = -1
  tracker = ""
  job_name = ""
  task_index = -1

  opts, args = getopt.getopt(argv,
                             's:c:t:j:ti:',
                             ['server_count=', 'client_count=', 'tracker=',
                              'job_name=', 'task_index='])
  for opt, arg in opts:
    if opt in ('-s', '--server_count'):
      server_count = int(arg)
    elif opt in ('-c', '--client_count'):
      client_count = int(arg)
    elif opt in ('-t', '--tracker'):
      tracker = arg
    elif opt in ('-j', '--job_name'):
      job_name = arg
    elif opt in ('-ti', '--task_index'):
      task_index = int(arg)
    else:
      pass

  g = gl.Graph()

  g.node(os.path.join(cur_path, "data/user"),
         node_type="user", decoder=gl.Decoder(weighted=True)) \
    .node(os.path.join(cur_path, "data/item"),
          node_type="item", decoder=gl.Decoder(attr_types=['string', 'int', 'float', 'float', 'string'])) \
    .edge(os.path.join(cur_path, "data/u-i"),
          edge_type=("user", "item", "buy"), decoder=gl.Decoder(weighted=True), directed=False) \
    .node(os.path.join(cur_path, "data/entity"),
          node_type="entity", decoder=gl.Decoder(attr_types=['float', 'float', 'float', 'float'], labeled=True)) \
    .edge(os.path.join(cur_path, "data/relation"),
          edge_type=("entity", "entity", "relation"), decoder=gl.Decoder(weighted=True), directed=False) \
    .edge(os.path.join(cur_path, "data/relation"),
          edge_type=("cond_node", "cond_node", "cond_edge"), decoder=gl.Decoder(weighted=True), directed=True) \
    .node(os.path.join(cur_path, "data/cond_node"),
          node_type="cond_node", decoder=gl.Decoder(attr_types=['int','int','float','string'], weighted=True))

  cluster={"server_count": server_count, "client_count": client_count, "tracker":tracker}
  g.init(cluster=cluster, job_name=job_name, task_index=task_index)

  if job_name == "server":
    print("Server {} started.".format(task_index))
    g.wait_for_close()

  if job_name == "client":
    print("Client {} started.".format(task_index))

    test_node_iterate(g)
    test_edge_iterate(g)
    test_truncated_full_edge_sample(g)
    test_conditional_negtaive_sample(g)
    test_get_stats(g)

    g.close()
    print("Client {} stopped.".format(task_index))

if __name__ == "__main__":
  main(sys.argv[1:])
