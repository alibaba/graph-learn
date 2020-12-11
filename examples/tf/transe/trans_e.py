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
import numpy as np
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf


class TransE(gl.LearningBasedModel):
  """
  Args:
    graph: Initialized gl.Graph object.
    batch_size: Batch size for training set.
    hidden_dim: Hidden dimension.
    neg_num: The number of negative samples for each node.
    s2h: Set True if using string2hash for lookup embedding.
    entity_num: Number of entities.
    relation_num: Number of relations.
    margin: Margin used in loss
    ps_hosts: Set when using distributed mode.
  """
  def __init__(self,
               graph,
               neg_num,
               batch_size,
               margin,
               entity_num,
               relation_num,
               hidden_dim,
               s2h=False,
               ps_hosts=None):
    super(TransE, self).__init__(graph,
                                 batch_size)
    self.margin = margin
    self.entity_num = entity_num
    self.relation_num = relation_num
    self.hidden_dim = hidden_dim
    self.s2h = s2h
    self.neg_num = neg_num
    self.ps_hosts = ps_hosts

    self.node_ego_spec = gl.EgoSpec(gl.FeatureSpec(1, 0))
    # continuous features as edge ids.
    self.edge_ego_spec = gl.EgoSpec(gl.FeatureSpec(1, 0))
    self.encoders = self._encoders()

  def _sample_seed(self):
    """
    for TransE, emit "e": <h, r, t> or <src, edge, dst>
    """
    # TODO: in transE, relation is vertex but is also edge,
    #  it appears in the edge table as a string weight.
    return self.graph.E("hrt").shuffle(traverse=True)\
      .batch(self.batch_size).values()

  def _entity_sample_seed(self):
    return self.graph.V("entity").batch(self.batch_size).values()

  def _relation_sample_seed(self):
    return self.graph.V("relation").batch(self.batch_size).values()

  def _positive_sample(self, t):
    if isinstance(t, gl.Nodes):
      # faked edge used to save embedding.
      return gl.Edges(t.ids, 'entity',
                      t.ids, 'entity',
                      'relation', graph=self.graph)
    else:
      return t

  def _negative_sample(self, t):
    """
    In this example, we use a proportion near 0.5 to corrupt heads or
    tails correspondingly, to build negative examples. Dynamic proportion
    can also be used since _negative_sample involves operations only on python
    objects rather than tf tensors. Notice the shape of negative example
    is different from that in positive example, so that a tile-style operation
    is needed. Finally we concat the Edges corrupted by both pipelines and
    produce the final negative samples.
    """
    #TODO: Refactor.
    def duplicate(a, times):
      return np.tile(np.expand_dims(a, -1), [1, times])

    def concat_edge(a, b):
      return gl.Edges(np.concatenate([a.src_ids, b.src_ids], -1), a.src_nodes.type,
                      np.concatenate([a.dst_ids, b.dst_ids], -1), a.dst_nodes.type,
                      a.edge_type,
                      edge_ids=np.concatenate([a.edge_ids, b.edge_ids], -1),
                      int_attrs=np.concatenate([a.int_attrs, b.int_attrs], 1),
                      graph=self.graph)

    n_corrupt_tail = self.neg_num // 2
    n_corrupt_head = self.neg_num - n_corrupt_tail
    res_corrupt_tail = \
        self.graph.E("hrt", feed=t).alias('r')\
            .each(lambda e: (e.outV().alias('h'),
                             e.inV().outNeg('hrt').sample(n_corrupt_tail).by('random').alias('nt')))\
            .emit(lambda x: gl.Edges(
                duplicate(x['h'].ids, n_corrupt_tail), x['h'].type,
                x['nt'].ids, x['nt'].type,
                x['r'].edge_type,
                edge_ids = duplicate(np.zeros(x['r'].shape[0]), n_corrupt_tail),
                int_attrs = duplicate(x['r'].int_attrs[:, 0], n_corrupt_tail),
                graph = self.graph))
    res_corrupt_head = \
        self.graph.E("hrt", feed=t).alias('r')\
            .each(lambda e: (e.outV().outNeg('hrt').sample(n_corrupt_head).by('random').alias('nh'),
                          e.inV().alias('t')))\
            .emit(lambda x: gl.Edges(
                x['nh'].ids, x['nh'].type,
                duplicate(x['t'].ids, n_corrupt_head), x['t'].type,
                x['r'].edge_type,
                edge_ids = duplicate(np.zeros(x['r'].shape[0]), n_corrupt_head),
                int_attrs = duplicate(x['r'].int_attrs[:, 0], n_corrupt_head),
                graph = self.graph))
    res = concat_edge(res_corrupt_tail, res_corrupt_head)
    return res

  def _receptive_fn(self, nodes):
    """TransE is a model which requires no receptive field of entities.
    So only a straight forward fn is needed.
    """
    # in fact, pos_node is edge.
    return gl.EgoGraph(nodes, layers=None)

  def _encoders(self):
    initializer = tf.glorot_normal_initializer()
    entity_encoder = gl.encoders.LookupEncoder(self.entity_num,
                                               self.hidden_dim,
                                               str2hash=self.s2h,
                                               ps_hosts=self.ps_hosts,
                                               init=initializer,
                                               name='entity_encoder')
    relation_encoder = gl.encoders.LookupEncoder(self.relation_num,
                                                 self.hidden_dim,
                                                 str2hash=self.s2h,
                                                 ps_hosts=self.ps_hosts,
                                                 init=initializer,
                                                 use_edge=True,
                                                 name='relation_encoder')
    return {"src": entity_encoder, "edge": relation_encoder, "dst": entity_encoder}

  def _unsupervised_loss_with_edge(self, pos_src_emb, pos_edge_emb, neg_src_emb,
                                   pos_dst_emb, neg_edge_emb, neg_dst_emb):
    return gl.triplet_loss(
        pos_src_emb, pos_edge_emb, neg_src_emb,
        pos_dst_emb, neg_edge_emb, neg_dst_emb,
        self.margin, self.neg_num)

  def build(self):
    ego_flow = gl.EgoFlow(self._sample_seed,
                          self._positive_sample,
                          self._receptive_fn,
                          self.node_ego_spec,
                          dst_ego_spec=self.node_ego_spec,
                          edge_ego_spec=self.edge_ego_spec,
                          negative_sample=self._negative_sample)
    iterator = ego_flow.iterator
    pos_src_ego_tensor = ego_flow.pos_src_ego_tensor
    pos_dst_ego_tensor = ego_flow.pos_dst_ego_tensor
    neg_src_ego_tensor = ego_flow.neg_src_ego_tensor
    neg_dst_ego_tensor = ego_flow.neg_dst_ego_tensor
    pos_edge_ego_tensor = ego_flow.pos_edge_ego_tensor
    neg_edge_ego_tensor = ego_flow.neg_edge_ego_tensor

    pos_src_emb = self.encoders['src'].encode(pos_src_ego_tensor)
    pos_dst_emb = self.encoders['dst'].encode(pos_dst_ego_tensor)
    neg_src_emb = self.encoders['src'].encode(neg_src_ego_tensor)
    neg_dst_emb = self.encoders['dst'].encode(neg_dst_ego_tensor)
    pos_edge_emb = self.encoders['edge'].encode(pos_edge_ego_tensor)
    neg_edge_emb = self.encoders['edge'].encode(neg_edge_ego_tensor)
    loss = self._unsupervised_loss_with_edge(pos_src_emb, pos_edge_emb, neg_src_emb,
                                             pos_dst_emb, neg_edge_emb, neg_dst_emb)
    return loss[0], iterator

  def node_embedding(self, type='entity'):
    """Return embeddings for saving"""
    if type == 'entity':
      # for save entity embeddings
      ego_flow = gl.EgoFlow(self._entity_sample_seed,
                            self._positive_sample,
                            self._receptive_fn,
                            self.node_ego_spec)
      iterator = ego_flow.iterator
      ego_tensor = ego_flow.pos_src_ego_tensor
      emb = self.encoders['src'].encode(ego_tensor)
      ids = ego_tensor.src.ids
    else:
      # for save relation embeddings
      ego_flow = gl.EgoFlow(self._relation_sample_seed,
                            self._positive_sample,
                            self._receptive_fn,
                            self.node_ego_spec)
      iterator = ego_flow.iterator
      ego_tensor = ego_flow.pos_src_ego_tensor
      emb = self.encoders['edge'].encode(ego_tensor)
      ids = ego_tensor.src.ids

    return ids, emb, iterator
