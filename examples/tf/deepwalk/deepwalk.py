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
"""class of DeepWalk model."""
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


class DeepWalk(gl.LearningBasedModel):
  """
  Args:
    graph: Initialized gl.Graph object.
    walk_len: Random walk length.
    window_size: Window size.
    node_count: Total numebr of nodes.
    batch_size: Batch size for training set.
    hidden_dim: Hidden dimension.
    neg_num: The number of negative samples for each node.
    s2h: Set it to True if using string2hash.
    ps_hosts: Set when running in distributed mode.
    temperature: Softmax temperature.
    node_type: User defined node type name.
    edge_type: User defined edge type name.
  """
  def __init__(self,
               graph,
               walk_len,
               window_size,
               node_count,
               hidden_dim,
               neg_num,
               batch_size,
               s2h=False,
               ps_hosts=None,
               temperature=1.0,
               node_type='item',
               edge_type='relation'):
    super(DeepWalk, self).__init__(graph,
                                   batch_size)
    self.walk_len = walk_len
    self.window_size = window_size
    self.node_count = node_count
    self.hidden_dim = hidden_dim
    self.neg_num = neg_num
    self.s2h = s2h
    self.ps_hosts=ps_hosts
    self.temperature=temperature
    self.node_type = node_type
    self.edge_type = edge_type

    # construct EgoSpecs.
    self.ego_spec = gl.EgoSpec(gl.FeatureSpec(0, 0))
    # encoders.
    self.encoders = self._encoders()

  def _sample_seed(self, mode='train'):
    return self.graph.V(self.node_type).batch(self.batch_size).values()

  def _positive_sample(self, t):
    path = self.graph.V(self.node_type, feed=t)\
        .repeat(lambda v: v.outV(self.edge_type).sample(1).by('random'),
               self.walk_len - 1)\
        .emit(lambda x: [x[i].ids.reshape([-1])
                         for i in range(self.walk_len)])
    src_ids, dst_ids = gl.gen_pair(path,
                                   self.window_size,
                                   self.window_size)
    return gl.Edges(src_ids, self.node_type,
                    dst_ids, self.node_type,
                    self.edge_type, graph=self.graph)

  def _positive_sample_save(self, t):
    # fake edge for saving node embeddings.
    return gl.Edges(t.ids, self.node_type,
                    t.ids, self.node_type,
                    self.edge_type, graph=self.graph)

  def _negative_sample(self, t):
    return self.graph.V(self.node_type, feed=t.src_nodes)\
      .outNeg(self.edge_type)\
      .sample(self.neg_num).by("random").emit(lambda x: x[1])

  def _receptive_fn(self, nodes):
    return gl.EgoGraph(nodes, layers=None)

  def _encoders(self):
    src_encoder = gl.encoders.LookupEncoder(self.node_count,
                                            self.hidden_dim,
                                            init=None,
                                            str2hash=self.s2h,
                                            ps_hosts=self.ps_hosts,
                                            name='node_encoder')
    dst_encoder = gl.encoders.LookupEncoder(self.node_count,
                                            self.hidden_dim,
                                            init=tf.zeros_initializer(),
                                            str2hash=self.s2h,
                                            ps_hosts=self.ps_hosts,
                                            name='context_encoder')
    return {"src": src_encoder, "edge": None, "dst": dst_encoder}

  def _unsupervised_loss(self, src_emb, pos_dst_emb):
    return gl.sampled_softmax_loss(
        src_emb / self.temperature,
        pos_dst_emb,
        self.neg_num * self.batch_size,
        self.encoders['dst'].emb_table,
        self.encoders['dst'].bias_table,
        self.encoders['dst'].num,
        self.s2h)

  def build(self):
    self.ego_flow = gl.EgoFlow(self._sample_seed,
                              self._positive_sample,
                              self._receptive_fn,
                              self.ego_spec,
                              dst_ego_spec=self.ego_spec, # homo graph.
                              negative_sample=None) # use sampled softmax loss.
    self.iterator = self.ego_flow.iterator
    self.pos_src_ego_tensor = self.ego_flow.pos_src_ego_tensor
    self.pos_dst_ego_tensor = self.ego_flow.pos_dst_ego_tensor
    pos_src_emb = self.encoders['src'].encode(self.pos_src_ego_tensor)
    pos_dst_emb = self.pos_dst_ego_tensor.src.ids  # use sampled softmax

    self.loss = self._unsupervised_loss(pos_src_emb, pos_dst_emb)
    self.loss = self.loss[0]

    return self.loss, self.iterator


  def node_embedding(self, type):
    node_ego_flow = gl.EgoFlow(self._sample_seed,
                               self._positive_sample_save,
                               self._receptive_fn,
                               self.ego_spec)
    iterator = node_ego_flow.iterator
    ego_tensor = node_ego_flow.pos_src_ego_tensor
    src_emb = self.encoders['src'].encode(ego_tensor)
    src_ids = node_ego_flow.pos_src_ego_tensor.src.ids
    return src_ids, src_emb, iterator
