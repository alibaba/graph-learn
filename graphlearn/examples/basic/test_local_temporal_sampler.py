# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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

import graphlearn as gl

def main(argv):
  cur_path = sys.path[0]
  gl.set_tape_capacity(1)
  gl.set_padding_mode(0)
  gl.set_default_neighbor_id(-1)

  g = gl.Graph()
  g.node(os.path.join(cur_path, "data/user"),
         node_type="user", decoder=gl.Decoder(weighted=True)) \
    .edge(os.path.join(cur_path, "data/u-u"),
          edge_type=("user", "user", "interaction"),
          decoder=gl.Decoder(weighted=True, timestamped=True))
  g.init()

  events = g.E("interaction").batch(2).alias("event")
  srcV = events.outV().alias('src')
  dstV = events.inV().alias('pos_dst')
  negV = srcV.outNeg("interaction").sample(1).by("random").alias("neg_dst")
  srcV_nbr = srcV.outE("interaction").sample(3).by("topk").alias("src_nbr")
  dstV_nbr = dstV.outE("interaction").sample(3).by("topk").alias("dst_nbr")
  negV_nbr = negV.outE("interaction").sample(3).by("topk").alias("neg_nbr")

  query = events.values()

  ds = gl.Dataset(query, 1)

  while True:
    try:
      event = ds.next()
      print(event["event"].src_ids, event["event"].dst_ids, event["event"].timestamps)
      print(event["dst_nbr"].dst_ids, event["dst_nbr"].timestamps)
    except gl.OutOfRangeError:
      break

  g.close()


if __name__ == "__main__":
  main(sys.argv[1:])
