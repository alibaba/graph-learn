from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getopt
import os
import sys

import graphlearn as gl
import numpy as np
import time


def main(argv):
  cur_path = sys.path[0]

  cluster = ""
  job_name = ""
  task_index = 0
  mode = 0

  opts, args = getopt.getopt(argv, 'c:j:t:', ['cluster=', 'job_name=', 'task_index=', 'mode='])
  for opt, arg in opts:
    if opt in ('-c', '--cluster'):
      cluster = arg
    elif opt in ('-j', '--job_name'):
      job_name = arg
    elif opt in ('-t', '--task_index'):
      task_index = int(arg)
    elif opt in ('-m', '--mode'):
      mode = int(arg)
    else:
      pass

  gl.set_tracker_mode(mode)

  g = gl.Graph()

  g.node(os.path.join(cur_path, "data/user"),
         node_type="user", decoder=gl.Decoder(weighted=True)) \
    .node(os.path.join(cur_path, "data/item"),
          node_type="item", decoder=gl.Decoder(attr_types=['string', 'int', 'float', 'float', 'string'])) \
    .edge(os.path.join(cur_path, "data/u-i"),
          edge_type=("user", "item", "buy"), decoder=gl.Decoder(weighted=True))

  g.init(cluster=cluster, job_name=job_name, task_index=task_index)

  if job_name == "server":
    print("Server {} started.".format(task_index))
    g.wait_for_close()

  if job_name == "client":
    print("Client {} started.".format(task_index))
    q = g.V("user").batch(10).values()
    for i in range(3):
      while True:
        try:
          print(g.run(q).ids)
        except gl.OutOfRangeError:
          print("Out of range......")
          break

    q = g.E("buy").batch(10).values()
    for i in range(3):
      while True:
        try:
          print(g.run(q).dst_ids)
        except gl.OutOfRangeError:
          print("Out of range......")
          break
    g.close()

if __name__ == "__main__":
  main(sys.argv[1:])