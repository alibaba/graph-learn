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


class BipartiteGraphSage(gl.LearningBasedModel):
  """
  Args:
    graph: Initialized gl.Graph object.
    batch_size: Batch size for training set.
    hidden_dim: Hidden dimension.
    output_dim: The final output embedding dimension.
    hops_num: Number of hops to perform neighbor sampling.
    u_neighs_num: A list indicates number of u neighbors to sample in
    each hop, with the format [u_hop1_num, u_hop2_num, ...].
    i_neighs_num: A list indicates number of i neighbors to sample in
    each hop,  with the format [i_hop1_num, i_hop2_num, ...].
    u_features_num: User features dimension.
    u_categorical_attrs_desc: A dict indicates u discrete features,
    with the format
    {feature column index : [name, max number, embedding dimension]}.
    i_features_num: Item features dimension.
    i_categorical_attrs_desc: A dict indicates i discrete features, with the format
    {feature column index : [name, max number, embedding dimension]}.
    neg_num: The number of negative samples for each node.
    use_input_bn: Used by gl.WideNDeepEncoder, set it to False if not using batch norm.
    act: Activation function for gl.WideNDeepEncoder.
    agg_type: Aggregation type. 'gcn', 'mean' or 'sum'.
    need_dense: Set False if not adding a dense layer in Wide&Deep Encoder.
    in_drop_rate: Dropout ratio for input embedded data.
  """
  def __init__(self,
               graph,
               batch_size,
               hidden_dim,
               output_dim,
               hops_num,
               u_neighs_num,
               i_neighs_num,
               u_features_num=0,
               u_categorical_attrs_desc='',
               i_features_num=0,
               i_categorical_attrs_desc='',
               neg_num=10,
               use_input_bn=True,
               act=tf.nn.leaky_relu,
               agg_type='gcn',
               need_dense=True,
               in_drop_rate=0.5,
               ps_hosts=None):
    super(BipartiteGraphSage, self).__init__(graph,
                                             batch_size)
    self.hops_num = hops_num
    self.neg_num = neg_num
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.u_neighs_num = u_neighs_num
    self.i_neighs_num = i_neighs_num
    self.u_features_num = u_features_num
    self.u_categorical_attrs_desc = u_categorical_attrs_desc
    self.i_features_num = i_features_num
    self.i_categorical_attrs_desc = i_categorical_attrs_desc

    self.use_input_bn = use_input_bn
    self.act = act
    self.agg_type = agg_type
    self.need_dense = need_dense
    self.in_drop_rate = in_drop_rate
    self.ps_hosts = ps_hosts

    # construct EgoSpecs.
    u_categorical_attrs_num = len(u_categorical_attrs_desc)
    u_continuous_attrs_num = u_features_num - u_categorical_attrs_num
    i_categorical_attrs_num = len(i_categorical_attrs_desc)
    i_continuous_attrs_num = i_features_num - i_categorical_attrs_num

    u_spec = gl.FeatureSpec(u_continuous_attrs_num,
                            u_categorical_attrs_num)
    i_spec = gl.FeatureSpec(i_continuous_attrs_num,
                            i_categorical_attrs_num)
    u_hop_spec = gl.HopSpec(i_spec)
    i_hop_spec = gl.HopSpec(u_spec)
    u_hops_spec = [] # u-i-u-i-u...
    i_hops_spec = [] # i-u-i-u-i...
    for i in range(self.hops_num):
      if (i % 2) == 0:
        u_hops_spec.append(u_hop_spec)
        i_hops_spec.append(i_hop_spec)
      else:
        u_hops_spec.append(i_hop_spec)
        i_hops_spec.append(u_hop_spec)

    self.u_ego_spec = gl.EgoSpec(u_spec, hops_spec=u_hops_spec)
    self.i_ego_spec = gl.EgoSpec(i_spec, hops_spec=i_hops_spec)

    # encoders.
    self.encoders = self._encoders()

  def _sample_seed(self):
    return self.graph.E("u-i").shuffle(traverse=True).batch(self.batch_size).values()

  def _u_sample_seed(self):
    return self.graph.V('u').batch(self.batch_size).values()

  def _i_sample_seed(self):
    return self.graph.V('i').batch(self.batch_size).values()

  def _positive_sample(self, t):
    if isinstance(t, gl.Nodes):
      # faked edge used to save embedding.
      return gl.Edges(t.ids, t.type, t.ids, t.type, 'fake', graph=self.graph)
    else:
      return t

  def _negative_sample(self, t):
    return self.graph.E("u-i", feed=t).outV().alias('vSrc')\
      .outNeg("u-i").sample(self.neg_num).by("random").alias('vNeg')\
      .emit(lambda x: x['vNeg'])

  #TODO: support multi hops.
  def _receptive_fn(self, nodes):
    if nodes.type == 'u':
      neighs_num = self.u_neighs_num
      return self.graph.V(nodes.type, nodes).alias('v').outV("u-i")\
        .sample(neighs_num[0]).by("random").alias('v1')\
        .emit(lambda x: gl.EgoGraph(x['v'], [gl.Layer(nodes=x['v1'])]))
    else:
      neighs_num = self.i_neighs_num
      return self.graph.V(nodes.type, nodes).alias('v').inV("u-i")\
        .sample(neighs_num[0]).by("random").alias('v1')\
        .emit(lambda x: gl.EgoGraph(x['v'], [gl.Layer(nodes=x['v1'])]))

  def _unsupervised_loss(self, src_emb, pos_dst_emb, neg_dst_emb):
    return gl.sigmoid_cross_entropy_loss(src_emb, pos_dst_emb, neg_dst_emb, 'dot')

  def _encoders(self):
    self.in_drop = tf.placeholder(tf.float32, shape=None, name='dropout_ph')

    u_feature_encoder = gl.encoders.WideNDeepEncoder(
        self.u_categorical_attrs_desc,
        self.u_features_num,
        self.hidden_dim,
        use_input_bn=self.use_input_bn,
        act=self.act,
        need_dense=self.need_dense,
        ps_hosts=self.ps_hosts,
        name='u_feat_encoder')
    i_feature_encoder = gl.encoders.WideNDeepEncoder(
        self.i_categorical_attrs_desc,
        self.i_features_num,
        self.hidden_dim,
        use_input_bn=self.use_input_bn,
        act=self.act,
        need_dense=self.need_dense,
        ps_hosts=self.ps_hosts,
        name='i_feat_encoder')

    depth = self.hops_num
    u_feature_encoders = []
    i_feature_encoders = []
    for i in range(depth + 1):
      if (i % 2) == 0:
        u_feature_encoders.append(u_feature_encoder)
        i_feature_encoders.append(i_feature_encoder)
      else:
        u_feature_encoders.append(i_feature_encoder)
        i_feature_encoders.append(u_feature_encoder)

    u_conv_layers = []
    i_conv_layers = []
    # for hidden layer
    for i in range(0, depth-1):
      u_conv_layers.append(gl.layers.GraphSageConv(i,
                                                   self.hidden_dim,
                                                   self.hidden_dim,
                                                   self.agg_type,
                                                   name='u'))
      i_conv_layers.append(gl.layers.GraphSageConv(i,
                                                   self.hidden_dim,
                                                   self.hidden_dim,
                                                   self.agg_type,
                                                   name='i'))
    # for output layer
    u_conv_layers.append(gl.layers.GraphSageConv(depth-1,
                                                 self.hidden_dim,
                                                 self.output_dim,
                                                 self.agg_type,
                                                 act=None,
                                                 name='u'))
    i_conv_layers.append(gl.layers.GraphSageConv(depth-1,
                                                 self.hidden_dim,
                                                 self.output_dim,
                                                 self.agg_type,
                                                 act=None,
                                                 name='i'))

    u_encoder = gl.encoders.EgoGraphEncoder(u_feature_encoders,
                                            u_conv_layers,
                                            self.u_neighs_num,
                                            dropout=self.in_drop)
    i_encoder = gl.encoders.EgoGraphEncoder(i_feature_encoders,
                                            i_conv_layers,
                                            self.i_neighs_num,
                                            dropout=self.in_drop)

    return {"src": u_encoder, "edge": None, "dst": i_encoder}


  def build(self):
    ego_flow = gl.EgoFlow(self._sample_seed,
                          self._positive_sample,
                          self._receptive_fn,
                          self.u_ego_spec,
                          dst_ego_spec=self.i_ego_spec,
                          negative_sample=self._negative_sample)
    iterator = ego_flow.iterator
    pos_src_ego_tensor = ego_flow.pos_src_ego_tensor
    pos_dst_ego_tensor = ego_flow.pos_dst_ego_tensor
    neg_dst_ego_tensor = ego_flow.neg_dst_ego_tensor

    src_emb = self.encoders['src'].encode(pos_src_ego_tensor)
    pos_dst_emb = self.encoders['dst'].encode(pos_dst_ego_tensor)
    neg_dst_emb = self.encoders['dst'].encode(neg_dst_ego_tensor)
    loss = self._unsupervised_loss(src_emb, pos_dst_emb, neg_dst_emb)
    return loss[0], iterator


  def node_embedding(self, type='u'):
    """Return embeddings for saving"""
    if type == 'u':
      # for save u embeddings
      ego_flow = gl.EgoFlow(self._u_sample_seed,
                            self._positive_sample,
                            self._receptive_fn,
                            self.u_ego_spec)
      iterator = ego_flow.iterator
      ego_tensor = ego_flow.pos_src_ego_tensor
      emb = self.encoders['src'].encode(ego_tensor)
      ids = ego_tensor.src.ids
    else:
      # for save i embeddings
      ego_flow = gl.EgoFlow(self._i_sample_seed,
                            self._positive_sample,
                            self._receptive_fn,
                            self.i_ego_spec)
      iterator = ego_flow.iterator
      ego_tensor = ego_flow.pos_src_ego_tensor
      emb = self.encoders['dst'].encode(ego_tensor)
      ids = ego_tensor.src.ids

    return ids, emb, iterator

  def feed_training_args(self):
    return {self.in_drop: self.in_drop_rate}

  def feed_evaluation_args(self):
    return {self.in_drop: 0.0}
