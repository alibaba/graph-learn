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
import sys
import shutil

import graphlearn as gl
from query_examples import *

cur_path = sys.path[0]

def gen_data():
  os.mkdir(os.path.join(cur_path, "tmp_data"))
  with open(os.path.join(cur_path, "tmp_data/user1"), 'w') as f:
    for i in range(50):
      s = '%d\t%f\n' % (i, i / 10.0)
      f.write(s)
  with open(os.path.join(cur_path, "tmp_data/user2"), 'w') as f:
    for i in range(50):
      s = '%d\t%f\n' % (i+50, i)
      f.write(s)

def main():
  # This test requires downloading the Hadoop distribution to run.
  # Please ensure export $JAVA_HOME , $HADOOP_HOME and
  # export JRE_HOME=${JAVA_HOME}/jre
  # export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
  # export CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}
  # export PATH=${JAVA_HOME}/bin:${HADOOP_HOME}bin:${HADOOP_HOME}/sbin:$PATH
  # and also you should source env.sh before testing.
  gen_data()
  g = gl.Graph()
  # To test against the real distributed cluster, use following path:
  # hdfs://cluster/test/tmp/dir/
  g.node("file://tmp" + os.path.join(cur_path, "tmp_data/"),
         node_type="user", decoder=gl.Decoder(weighted=True))
  g.init()
  print(g.get_stats())
  assert(g.get_stats()['user'][0]==100)
  sampler = g.node_sampler('user', 10)
  for i in range(20):
    try:
      nodes = sampler.get()
      print(nodes.ids, nodes.weights)
    except gl.OutOfRangeError:
      break
  g.close()
  shutil.rmtree(os.path.join(cur_path, "tmp_data"))

if __name__ == "__main__":
  main()
