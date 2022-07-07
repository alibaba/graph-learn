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

import graphlearn as gl

def test_node_iterate(graph, local=False):
  """Iterate users, sample 2 hops with path user-(buy)-item-(buy_reverse)-user.
  """
  query = graph.V("user").batch(32).shuffle(traverse=True).alias("src") \
          .outV("buy").sample(5).by("edge_weight").alias("hop1") \
          .inE("buy").sample(2).by("random").alias("hop1-hop2") \
          .inV().alias("hop2") \
          .values()

  ds = gl.Dataset(query)
  epoch = 2
  for i in range(epoch):
    step = 0
    while True:
      try:
        res = ds.next()
        step += 1

        src_nodes = res["src"]
        hop1_nodes = res["hop1"]
        hop1_hop2_edges = res["hop1-hop2"]
        hop2_nodes = res["hop2"]

        assert isinstance(src_nodes, gl.Nodes)
        assert isinstance(hop1_nodes, gl.Nodes)
        assert isinstance(hop1_hop2_edges, gl.Edges)
        assert isinstance(hop2_nodes, gl.Nodes)

        assert src_nodes.type == "user"
        assert hop1_nodes.type == "item"
        assert hop1_hop2_edges.type == ("item", "user", "buy_reverse")
        assert hop1_hop2_edges.edge_type == "buy_reverse"
        assert hop2_nodes.type == "user"

        if local and step == 100 // 32 + 1:  # total user nodes count is 100
          assert tuple(src_nodes.ids.shape) == (100 % 32, )
          assert tuple(hop1_nodes.ids.shape) == (100 % 32, 5)
          assert tuple(hop1_hop2_edges.src_ids.shape) == (100 % 32 * 5, 2)
          assert tuple(hop1_hop2_edges.dst_ids.shape) == (100 % 32 * 5, 2)
          assert tuple(hop2_nodes.ids.shape) == (100 % 32 * 5, 2)

          assert tuple(hop1_nodes.float_attrs.shape) == (100 % 32, 5, 2)  # 2 float attrs
          assert tuple(hop1_hop2_edges.weights.shape) == (100 % 32 * 5, 2)
          assert tuple(hop2_nodes.weights.shape) == (100 % 32 * 5, 2)
        elif local or step == 1:
          assert tuple(src_nodes.ids.shape) == (32, )
          assert tuple(hop1_nodes.ids.shape) == (32, 5)
          assert tuple(hop1_hop2_edges.src_ids.shape) == (32 * 5, 2)
          assert tuple(hop1_hop2_edges.dst_ids.shape) == (32 * 5, 2)
          assert tuple(hop2_nodes.ids.shape) == (32 * 5, 2)

          assert tuple(hop1_nodes.float_attrs.shape) == (32, 5, 2)  # 2 float attrs
          assert tuple(hop1_hop2_edges.weights.shape) == (32 * 5, 2)
          assert tuple(hop2_nodes.weights.shape) == (32 * 5, 2)
      except gl.OutOfRangeError:
        break

def test_edge_iterate(graph, local=False):
  """Iterate buy edges, sample hops of src and dst nodes.
    user-(buy)-item   (1) iterate edges
      |         |
    (buy)  (buy_reverse)
      |         |
    item       user   (2) sample neighbors of src and dst nodes. `
  """
  edges = graph.E("buy").batch(32).shuffle(traverse=True).alias("edges")
  src = edges.outV().alias("src")
  dst = edges.inV().alias("dst")
  neg = src.outNeg("buy").sample(2).by("in_degree").alias("neg")

  neg.inV("buy").sample(4).by("random").alias("neg_hop1")
  src.outE("buy").sample(5).by("random").alias("src_hop1_edges") \
     .inV().alias("src_hop1")
  dst.inE("buy").sample(3).by("edge_weight").alias("dst_hop1_edges") \
      .inV().alias("dst_hop1")

  query = edges.values()
  ds = gl.Dataset(query)
  epoch = 2
  for i in range(epoch):
    step = 0
    while True:
      try:
        res = ds.next()
        step += 1

        edges = res["edges"]
        src_nodes = res["src"]
        dst_nodes = res["dst"]
        neg_nodes = res["neg"]
        src_hop1_edges = res["src_hop1_edges"]
        src_hop1_nodes = res["src_hop1"]
        neg_hop1_nodes = res["neg_hop1"]
        dst_hop1_edges = res["dst_hop1_edges"]
        dst_hop1_nodes = res["dst_hop1"]

        assert edges.type == ("user", "item", "buy")
        assert src_nodes.type == "user"
        assert dst_nodes.type == "item"
        assert neg_nodes.type == "item"
        assert src_hop1_edges.type == ("user", "item", "buy")
        assert src_hop1_nodes.type == "item"
        assert neg_hop1_nodes.type == "user"
        assert dst_hop1_edges.type == ("item", "user", "buy_reverse")
        assert dst_hop1_nodes.type == "user"

        if local and step == 1000 // 32 + 1:  # total buy edges count is 1000
          assert tuple(neg_nodes.float_attrs.shape) == (1000 % 32, 2, 2)
          assert tuple(neg_hop1_nodes.weights.shape) == (1000 % 32 * 2, 4, )
          assert tuple(src_hop1_edges.weights.shape) == (1000 % 32, 5)
          assert tuple(src_hop1_nodes.float_attrs.shape) == (1000 % 32, 5, 2)
          assert tuple(dst_hop1_edges.weights.shape) == (1000 % 32, 3)
          assert tuple(dst_hop1_nodes.weights.shape) == (1000 % 32, 3)
        elif local or step == 1:
          assert tuple(neg_nodes.float_attrs.shape) == (32, 2, 2)
          assert tuple(neg_hop1_nodes.weights.shape) == (32 * 2, 4, )
          assert tuple(src_hop1_edges.weights.shape) == (32, 5)
          assert tuple(src_hop1_nodes.float_attrs.shape) == (32, 5, 2)
          assert tuple(dst_hop1_edges.weights.shape) == (32, 3)
          assert tuple(dst_hop1_nodes.weights.shape) == (32, 3)

        src_ids = list(dst_hop1_edges.src_ids.flatten())
        dst_ids = list(dst_hop1_edges.dst_ids.flatten())
        weights = list(dst_hop1_edges.weights.flatten())
        for src_id, dst_id, weight in zip(src_ids, dst_ids, weights):
          assert abs(0.1 * (src_id + dst_id) - weight) < 10 ** -6

        src_ids = list(src_hop1_edges.src_ids.flatten())
        dst_ids = list(src_hop1_edges.dst_ids.flatten())
        weights = list(src_hop1_edges.weights.flatten())
        for src_id, dst_id, weight in zip(src_ids, dst_ids, weights):
          assert abs(0.1 * (src_id + dst_id) - weight) < 10 ** -6

      except gl.OutOfRangeError:
        break

def test_truncated_full_edge_sample(graph):
  """Iterate buy edges, and sample full neighbors of the dst nodes.
  """
  edges = graph.E("buy").batch(3).shuffle(traverse=True).alias("edges")
  dst = edges.inV().alias("dst")
  dst.inE("buy").sample(200).by("full").alias("dst_hop1_edges") \
      .inV().alias("dst_hop1")
  ds = gl.Dataset(edges.values())
  step = 0
  while True:
      try:
        res = ds.next()
        step += 1
        dst_hop1_edges = res["dst_hop1_edges"]
        if step == 1:
          print(dst_hop1_edges.offsets)
        src_ids = list(dst_hop1_edges.src_ids.flatten())
        dst_ids = list(dst_hop1_edges.dst_ids.flatten())
        weights = list(dst_hop1_edges.weights.flatten())
        for src_id, dst_id, weight in zip(src_ids, dst_ids, weights):
          assert abs(0.1 * (src_id + dst_id) - weight) < 10 ** -6
      except gl.OutOfRangeError:
        break

def test_conditional_negtaive_sample(graph):
  """Negative sampling with condition.
  """
  condition = {"unique": False, "batch_share": True,
               "int_cols": [0,1],
               "int_props": [0.25,0.25],
               "str_cols": [0],
               "str_props": [0.5]}

  edges = graph.E("cond_edge").batch(4).shuffle(traverse=True).alias("edges")
  src = edges.outV().alias("src")
  dst = edges.inV().alias("dst")
  src.outNeg("cond_edge").sample(5).by("in_degree") \
     .where(target="dst", condition=condition).alias("neg") \
          .values()

  ds = gl.Dataset(edges.values())
  try:
    res = ds.next()
    src_ids = res["src"].ids
    dst_ids = res["dst"].ids
    neg_nodes = res["neg"]
    for i in range(src_ids.size):
      print('src_id:%d\tdst_id:%d' % (src_ids[i], dst_ids[i]))
      print('neg_id_1:%d\tint_0_attr:%d' %(neg_nodes.ids[i][0], neg_nodes.int_attrs[i][0][0]))
      print('neg_id_2:%d\tint_1_attr:%d' %(neg_nodes.ids[i][1], neg_nodes.int_attrs[i][1][1]))
      print('neg_id_3:%d\tstr_0_attr:%s' %(neg_nodes.ids[i][2], neg_nodes.string_attrs[i][2][0]))
      print('neg_id_4:%d\tstr_0_attr:%s\n' %(neg_nodes.ids[i][3], neg_nodes.string_attrs[i][3][0]))
  except gl.OutOfRangeError:
    print("OutOfRange...")

def test_get_stats(graph):
  print(graph.get_stats())