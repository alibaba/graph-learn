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

    gl.enable_actor()
    gl.set_actor_local_shard_count(4)

    # Step 1: Construct graph with data source.
    #   Edges:
    #     user<--(buy)-->item
    #     entity<--(relation)-->entity
    #     cond_node--(cond_edge)-->cond_node
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
    g.init()

    # Step 2: Describe the queries on graph.
    test_node_iterate(g, local=True)
    test_edge_iterate(g, local=True)

    # fixme(@Seventeen17): bug exists in this test
    # test_truncated_full_edge_sample(g)

    g.close()


if __name__ == "__main__":
    main(sys.argv[1:])
