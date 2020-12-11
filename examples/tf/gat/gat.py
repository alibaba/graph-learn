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
"""class for GAT implementation"""
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


class GAT(gl.LearningBasedModel):
  """
  Args:
    graph: Initialized gl.Graph object.
    output_dim: Output dimension.
    features_num: Input features dimension.
    num_heads: Number of attention heads.
    batch_size: Batch size for training set.
    val_batch_size: Batch size for validation set.
    test_batch_size: Batch size for test set.
    categorical_attrs_desc: A dict indicates discrete features, with the format
    {feature columun index : [name, max number, embedding dimension]}.
    hidden_dim: Hidden dimension.
    in_drop_rate: Dropout ratio for input data.
    attn_drop_rate: Dropout ratio for attention coefficients.
    hops_num: Number of hops to perform neighbor sampling.
    neighs_num: A list indicates number of neighbors to sample in each hop,
    with the format [hop1_num, hop2_num, ...].
    full_graph_mode: Set True if sample full graph in first iteration.
    node_type: User defined node type name.
    edge_type: User defined edge type name.
  """

  def __init__(self,
               graph,
               output_dim,
               features_num,
               num_heads,
               batch_size,
               val_batch_size=None,
               test_batch_size=None,
               categorical_attrs_desc='',
               hidden_dim=16,
               in_drop_rate=0,
               attn_drop_rate=0,
               hops_num=2,
               neighs_num=None,
               full_graph_mode=False,
               node_type='item',
               edge_type='relation'):
    super(GAT, self).__init__(graph,
                              batch_size,
                              full_graph_mode=full_graph_mode)
    self.features_num = features_num
    self.categorical_attrs_desc = categorical_attrs_desc
    self.hops_num = hops_num
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.in_drop_rate = in_drop_rate
    self.attn_drop_rate= attn_drop_rate
    self.output_dim = output_dim
    self.neighs_num = neighs_num
    self.val_batch_size = val_batch_size if val_batch_size else batch_size
    self.test_batch_size = test_batch_size if test_batch_size else batch_size
    self.node_type = node_type
    self.edge_type = edge_type

    # construct EgoSpecs.
    categorical_attrs_num = len(categorical_attrs_desc)
    continuous_attrs_num = features_num - categorical_attrs_num
    src_spec = gl.FeatureSpec(continuous_attrs_num,
                              categorical_attrs_num,
                              labeled = True)
    hop_spec = gl.HopSpec(gl.FeatureSpec(continuous_attrs_num, categorical_attrs_num),
                          sparse=(self.neighs_num is None))
    self.src_ego_spec = gl.EgoSpec(src_spec, hops_spec=[hop_spec] * self.hops_num)
    # encoders.
    self.encoders = self._encoders()

  def _sample_seed(self):
    return self.graph.V('train').batch(self.batch_size).values()

  def _val_sample_seed(self):
    return self.graph.V('val').batch(self.val_batch_size).values()

  def _test_sample_seed(self):
    return self.graph.V('test').batch(self.test_batch_size).values()

  def _positive_sample(self, t):
    return gl.Edges(t.ids, self.node_type,
                    t.ids, self.node_type,
                    self.edge_type, graph=self.graph)

  def _receptive_fn(self, nodes):
    alias = ['v' + str(i + 1) for i in range(self.hops_num)]
    # sample based
    if self.neighs_num:
      assert len(self.neighs_num) == self.hops_num
      sample_func = lambda v, params: v.outV(self.edge_type).sample(params).by('topk')
      return self.graph.V(nodes.type, feed=nodes).alias('v')\
          .repeat(sample_func, self.hops_num, params_list=self.neighs_num, alias_list=alias)\
          .emit(lambda x: gl.EgoGraph(x['v'], [gl.Layer(nodes=x[name]) for name in alias]))

    # full batch based
    sample_func = lambda v: v.outV(self.edge_type).sample().by('full')
    return self.graph.V(nodes.type, feed=nodes).alias('v')\
        .repeat(sample_func, self.hops_num, alias_list=alias)\
        .emit(lambda x: gl.EgoGraph(x['v'], [gl.Layer(nodes=x[name]) for name in alias]))

  def _encoders(self):
    """
    return a dict of encoder
    """
    self.in_drop = tf.placeholder(tf.float32, None, name='in_drop_ph')
    self.attn_drop = tf.placeholder(tf.float32, None, name='attn_drop_ph')

    depth = self.hops_num
    feature_encoders = [gl.encoders.IdentityEncoder()] * (depth + 1)
    conv_layers = []
    for i in range(depth):
        dim = self.output_dim if i == depth - 1 else self.hidden_dim
        concat = False if (i == depth - 1 and depth != 1) else True
        act = None if (i == depth - 1 and depth != 1) else tf.nn.relu
        conv_layers.append(gl.layers.MultiHeadGATConv(dim,
                                                      self.num_heads[i],
                                                      attn_drop=self.attn_drop,
                                                      concat=concat,
                                                      act=act))
    if self.neighs_num:
      encoder = gl.encoders.EgoGraphEncoder(feature_encoders,
                                            conv_layers,
                                            nbr_num_list=self.neighs_num,
                                            dropout=self.in_drop)
    else:
      encoder = gl.encoders.SparseEgoGraphEncoder(feature_encoders,
                                                  conv_layers,
                                                  dropout=self.in_drop)

    return {"src": encoder, "edge": None, "dst":None}

  def _accuracy(self, logits, labels):
    """Accuracy for supervised model.
    Args:
      logits: embeddings, 2D tensor with shape [batchsize, dimension]
      labels: 1D tensor with shape [batchsize]
    """
    indices = tf.math.argmax(logits, 1, output_type=tf.int32)
    correct = tf.reduce_sum(tf.cast(tf.math.equal(indices, labels), tf.float32))
    return correct / tf.cast(tf.shape(labels)[0], tf.float32)

  def build(self):
    ego_flow = gl.EgoFlow(self._sample_seed,
                          self._positive_sample,
                          self._receptive_fn,
                          self.src_ego_spec,
                          full_graph_mode=self.full_graph_mode)
    iterator = ego_flow.iterator
    self.pos_src_ego_tensor = ego_flow.pos_src_ego_tensor
    src_emb = self.encoders['src'].encode(self.pos_src_ego_tensor)
    labels = self.pos_src_ego_tensor.src.labels
    loss = self._supervised_loss(src_emb, labels)

    return loss, iterator

  def val_acc(self):
    val_ego_flow = gl.EgoFlow(self._val_sample_seed,
                              self._positive_sample,
                              self._receptive_fn,
                              self.src_ego_spec,
                              full_graph_mode=self.full_graph_mode)
    val_iterator = val_ego_flow.iterator
    val_pos_src_ego_tensor = val_ego_flow.pos_src_ego_tensor
    val_logits = self.encoders['src'].encode(val_pos_src_ego_tensor)
    val_labels = val_pos_src_ego_tensor.src.labels
    return self._accuracy(val_logits, val_labels), val_iterator

  def test_acc(self):
    test_ego_flow = gl.EgoFlow(self._test_sample_seed,
                               self._positive_sample,
                               self._receptive_fn,
                               self.src_ego_spec,
                               full_graph_mode=self.full_graph_mode)
    test_iterator = test_ego_flow.iterator
    test_pos_src_ego_tensor = test_ego_flow.pos_src_ego_tensor
    test_logits = self.encoders['src'].encode(test_pos_src_ego_tensor)
    test_labels = test_pos_src_ego_tensor.src.labels
    return self._accuracy(test_logits, test_labels), test_iterator

  def _supervised_loss(self, emb, label):
    return gl.softmax_cross_entropy_loss(emb, label)

  def feed_training_args(self):
    return {self.in_drop: self.in_drop_rate,
            self.attn_drop: self.attn_drop_rate}

  def feed_evaluation_args(self):
    return {self.in_drop: 0.0, self.attn_drop: 0.0}
