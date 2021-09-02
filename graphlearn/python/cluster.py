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
import json
import os
import sys

from graphlearn import pywrap_graphlearn as pywrap 

def gen_cluster_info_from_tf_config():
  tf_config_json = os.environ.get("TF_CONFIG", "{}")
  print("TF_CONFIG=", tf_config_json)
  tf_config = json.loads(tf_config_json)
  cluster = tf_config.get("cluster", {})
  if cluster is None:
    raise ValueError("TF_CONFIG cluster is empty")

  ps_hosts = []
  gl_hosts = []
  worker_hosts = []
  for key, value in cluster.items():
    if "ps" == key:
      ps_hosts = value
    elif "worker" == key:
      worker_hosts = value
    elif "graphlearn" == key:
      gl_hosts = value
  if gl_hosts is None:
    gl_hosts = ps_hosts

  task = tf_config.get("task", {})
  if task is None:
    raise ValueError("TF_CONFIG task is empty")

  task_index = task['index']
  job_name = task['type']
  return ps_hosts, gl_hosts, worker_hosts, job_name, task_index

def gen_cluster_info_from_launch_params():
  ps_hosts = ""
  worker_hosts = ""
  job_name = ""
  task_index = -1
  gl_count = 0
  task_count = 0

  argv = sys.argv[1:]
  opts, args = getopt.getopt(
      argv, 'p:w:j:t:a:tc',
      ['ps_hosts=', 'worker_hosts=', 'job_name=',
       'task_index=', 'aligraph_count=', "task_count="])
  for opt, arg in opts:
    if opt in ('-p', '--ps_hosts'):
      ps_hosts = arg
    elif opt in ('-w', '--worker_hosts'):
      worker_hosts = arg
    elif opt in ('-j', '--job_name'):
      job_name = arg
    elif opt in ('-t', '--task_index'):
      task_index = int(arg)
    elif opt in ('-a', '--aligraph_count'):
      gl_count = int(arg)
    else:
      pass

  ps_hosts = ps_hosts.split(',')
  worker_hosts = worker_hosts.split(',')
  if job_name == "aligraph":
    job_name = "graphlearn"
  return ps_hosts, gl_count, worker_hosts, job_name, task_index

def get_cluster():
  tracker_mode = pywrap.get_tracker_mode()
  if tracker_mode == pywrap.TrackerMode.RPC:
    ps_hosts, gl_hosts, worker_hosts, job_name, task_index = \
        gen_cluster_info_from_tf_config()
    worker_count = len(worker_hosts)
    gl_cluster = {"server": ",".join(gl_hosts), "client_count": worker_count}
  else:
    ps_hosts, gl_count, worker_hosts, job_name, task_index = \
        gen_cluster_info_from_launch_params()
    worker_count = len(worker_hosts)
    gl_cluster = {"server_count": gl_count, "client_count": worker_count}
  tf_cluster = {"ps": ps_hosts, "worker": worker_hosts}
  return gl_cluster, tf_cluster, job_name, int(task_index)
