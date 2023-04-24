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

import numpy as np
import graphlearn.python.nn.tf as tfg

from graphlearn.python.nn.data import Data
from graphlearn.python.nn.subgraph import SubGraph


class LabelProcessor(tfg.SubGraphProcessor):
  def __init__(self,
               addl_types_and_shapes=None,
               max_dist=4):
    super(LabelProcessor, self).__init__(
        addl_types_and_shapes=addl_types_and_shapes)
    self.max_dist = max_dist

  def process_func(self, subgraph):
    """`SubGraph` preprocessing function.

    Args:
      subgraph: `SubGraph` in numpy format.

    Returns:
      processed `SubGraph`.
    """
    edge_index = subgraph.edge_index
    data = np.ones(edge_index[1].shape, dtype=np.int32)
    dist = np.sum([subgraph.dist_to_src, subgraph.dist_to_dst], axis=0)
    dist[dist < 0] = self.max_dist + 1
    dist[dist > self.max_dist] = self.max_dist + 1
    nodes = subgraph.nodes
    data = Data(nodes.ids,
                ints=nodes.int_attrs,
                floats=nodes.float_attrs,
                strings=nodes.string_attrs)
    return SubGraph(edge_index, data, struct_label=dist)
