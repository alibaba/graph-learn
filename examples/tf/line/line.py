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

import graphlearn as gl
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf


class LINE(gl.LearningBasedModel):
  """
  Args:
    graph: Initialized gl.Graph object.
    node_count: Total numebr of nodes.
    batch_size: Batch size for training set.
    hidden_dim: Hidden dimension.
    neg_num: The number of negative samples for each node.
    s2h: Set True if using string2hash in embedding lookup.
    ps_hosts: Set when using distributed mode.
    proximity: Set to 'first_order' or 'Second_order'.
    node_type: User defined node type name.
    edge_type: User defined edge type name.
  """
  def __init__(self,
               graph,
               node_count,
               hidden_dim,
               neg_num,
               batch_size,
               s2h=False,
               ps_hosts=None,
               proximity='first_order',
               node_type='item',
               edge_type='relation'):
    super(LINE, self).__init__(graph, batch_size)
    self.node_count = node_count
    self.hidden_dim = hidden_dim
    self.neg_num = neg_num
    self.s2h = s2h
    self.proximity = proximity
    self.ps_hosts = ps_hosts
    self.node_type = node_type
    self.edge_type = edge_type

    # construct EgoSpecs.
    self.ego_spec = gl.EgoSpec(gl.FeatureSpec(0, 0))
    # encoders.
    self.encoders = self._encoders()

  def _sample_seed(self):
    return self.graph.E(self.edge_type).shuffle(traverse=True)\
        .batch(self.batch_size).values()

  def _save_node_sample_seed(self):
    return self.graph.V(self.node_type).batch(self.batch_size).values()

  def _positive_sample(self, t):
    if isinstance(t, gl.Edges):
      return t
    else:
      return gl.Edges(t.ids, self.node_type,
                      t.ids, self.node_type,
                      self.edge_type, graph=self.graph)

  def _negative_sample(self, t):
    return self.graph.V(self.node_type, feed=t.src_ids)\
        .alias('vSrc').outNeg(self.edge_type)\
        .sample(self.neg_num).by("in_degree").alias('vNeg')\
        .emit(lambda x: x['vNeg'])

  def _receptive_fn(self, nodes):
    return gl.EgoGraph(nodes, layers=None)

  def _encoders(self):
    src_encoder = gl.encoders.LookupEncoder(self.node_count,
                                            self.hidden_dim,
                                            init=None,
                                            str2hash=self.s2h,
                                            ps_hosts=self.ps_hosts,
                                            name='first_encoder')
    if self.proximity == 'first_order':
      dst_encoder = src_encoder
    elif self.proximity == 'second_order':
      dst_encoder = gl.encoders.LookupEncoder(self.node_count,
                                              self.hidden_dim,
                                              init=tf.zeros_initializer(),
                                              str2hash=self.s2h,
                                              ps_hosts=self.ps_hosts,
                                              name='second_encoder')
    else:
      raise Exception("no encoder implemented!")

    return {"src": src_encoder, "edge": None, "dst": dst_encoder}

  def _unsupervised_loss(self, src_emb, pos_dst_emb, neg_dst_emb):
    return gl.kl_loss(src_emb, pos_dst_emb, neg_dst_emb)

  def build(self):
    ego_flow = gl.EgoFlow(self._sample_seed,
                          self._positive_sample,
                          self._receptive_fn,
                          self.ego_spec,
                          dst_ego_spec=self.ego_spec, # homo graph.
                          negative_sample=self._negative_sample)
    iterator = ego_flow.iterator
    pos_src_ego_tensor = ego_flow.pos_src_ego_tensor
    pos_dst_ego_tensor = ego_flow.pos_dst_ego_tensor
    neg_dst_ego_tensor = ego_flow.neg_dst_ego_tensor
    pos_src_emb = self.encoders['src'].encode(pos_src_ego_tensor)
    pos_dst_emb = self.encoders['dst'].encode(pos_dst_ego_tensor)
    neg_dst_emb = self.encoders['dst'].encode(neg_dst_ego_tensor)

    loss = self._unsupervised_loss(pos_src_emb, pos_dst_emb, neg_dst_emb)
    loss = loss[0]

    return loss, iterator

  def node_embedding(self, type):
    node_ego_flow = gl.EgoFlow(self._save_node_sample_seed,
                               self._positive_sample,
                               self._receptive_fn,
                               self.ego_spec)
    iterator = node_ego_flow.iterator
    ego_tensor = node_ego_flow.pos_src_ego_tensor
    src_emb = self.encoders['src'].encode(ego_tensor)
    src_ids = ego_tensor.src.ids
    return src_ids, src_emb, iterator
