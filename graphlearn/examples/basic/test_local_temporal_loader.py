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
from tqdm import tqdm

import graphlearn as gl
import graphlearn.python.nn.pytorch as thg


def main(argv):
  cur_path = sys.path[0]
  gl.set_tape_capacity(1)

  g = gl.Graph()
  g.node(os.path.join(cur_path, "data/user"),
         node_type="user", decoder=gl.Decoder(weighted=True)) \
    .edge(os.path.join(cur_path, "data/u-u"),
          edge_type=("user", "user", "interaction"),
          decoder=gl.Decoder(weighted=True, timestamped=True))
  g.init()

  query = g.E("interaction").batch(1).alias("event").values()

  ds = thg.TemporalDataset(query, 1, event_name="event")
  dl = thg.TemporalDataLoader(ds)

  for idx, data in tqdm(enumerate(dl)):
    print("res:", idx, data)
    print(data.src)
    print(data.dst)
    print(data.msg)
    print(data.t)

  g.close()


if __name__ == "__main__":
  main(sys.argv[1:])
