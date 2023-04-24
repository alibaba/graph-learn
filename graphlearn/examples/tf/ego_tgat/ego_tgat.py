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

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf
import graphlearn.python.nn.tf as tfg
from graphlearn.python.nn.tf.config import conf
from graphlearn.python.nn.tf.module import Module

from graphlearn.python.nn.tf.layers.linear_layer import LinearLayer

class EgoTGAT(Module):
  def __init__(self,
               dims, # <in_dim, out_dim> pairs
               num_head=1,
               bn_func=None,
               act_func=tf.nn.relu,
               dropout=0.0,
               attn_dropout=0.0,
               **kwargs):
    """ `EgoTGAT` is TGAT model trained with ego based `TemporalGraph`.
    Args:
      dims: <in_dim, out_dim> pairs for each layer.
        For input layer, in_dim = <src_node_feature_dim + src_node_time_dim>,
                         out_dim = hidden_dim.
        For hidden layers, in_dim = hidden_dim + time_dim,
                           out_dim is hidden_dim,
        For output layer, in_dim = hidden_dim + time_dim,
                          out_dim = output_dim.
      num_head: The number of head of multi-head attention.
      attn_dropout: dropout rate in attention.
    """
    super(EgoTGAT, self).__init__()

    assert len(dims) > 1
    if isinstance(num_head, int):
      num_head = [num_head]
    self.layers = []
    for i in range(len(dims)):
      in_dim, out_dim = dims[i]
      conv = tfg.EgoGATConv("gat_conv_" + str(i),
                            in_dim=in_dim,
                            out_dim=out_dim,
                            num_head=num_head[i % len(num_head)],
                            attn_dropout=attn_dropout)
      layer = ConvLayer([conv] * (len(dims) - i))
      self.layers.append(layer)
    self.bn_func = bn_func
    self.active_func = act_func
    self.dropout = dropout

  def forward(self, graph):
    graph = graph.transform() # feature transformation of `TemporalGraph`

    # h^{0}
    h = [[graph.src, graph.src_t]] # [[[B, D_f], [B, D_t]]]
    for i in range(len(self.layers)):
      h.append([graph.hop_node(i), graph.hop_edge(i), graph.hop_t(i)])

    hops = graph.nbr_nums
    for i in range(len(self.layers) - 1):
      # h^{i}
      current_hops = hops if i == 0 else hops[:-i]
      h = self.layers[i].forward(h, current_hops)
      H = []
      for idx, x in enumerate(h):
        if self.bn_func is not None:
          x = self.bn_func(x)
        if self.active_func is not None:
          x = self.active_func(x)
        if self.dropout and conf.training:
          x = tf.nn.dropout(x, keep_prob=1-self.dropout)
        H.append([x, graph.src_t if idx < 1 else graph.hop_t(idx - 1)])
      h = H

    # The last layer
    h = self.layers[-1].forward(h, [hops[0]])
    assert len(h) == 1
    return h[0]

class MergeLayer(Module):
  def __init__(self,
               dim1, dim2, dim3, dim4,
               active_fn=tf.nn.relu):
    self.dim1 = dim1
    self.dim2 = dim2
    self.dim3 = dim3
    self.active_func = active_fn

    self.layer1 = LinearLayer("merge_layer1",
                              input_dim=dim1 + dim2, output_dim=dim3,
                              use_bias=True)
    self.layer2 = LinearLayer("merge_layer2",
                              input_dim=dim3, output_dim=dim4,
                              use_bias=True)

  def forward(self, x1, x2):
    x = tf.concat([x1, x2], axis=-1)
    x = self.layer1(x)
    x = self.active_func(x)
    x = self.layer2(x)
    return x


class ConvLayer(Module):
  def __init__(self, convs):
    super(ConvLayer, self).__init__()
    self.convs = convs

  def forward(self, x_list, expands):
    """
    x_list: [[n, e, t], [hop1_n, hop1_e, hop1_t], [hop2_n, hop2_e, hop2_t]]
    """
    assert len(self.convs) == (len(x_list) - 1)
    assert len(self.convs) == len(expands)

    rets = []
    for i in range(1, len(x_list)):
      x = x_list[i - 1]
      neighbors = x_list[i]
      ret = self.convs[i - 1](
        tf.concat([x[0], x[-1]], axis=-1), # src node feat and time emb
        tf.concat(neighbors, axis=-1), expands[i - 1])
      rets.append(ret)
    return rets

  def append(self, conv):
    self.convs.append(conv)

class LinkScorePredict(Module):
  def __init__(self, feat_num):
    self.affinity_score = MergeLayer(feat_num, feat_num, feat_num, 1)

  def forward(self, src_emb, pos_dst_emb, neg_dst_emb):
    pos_score = tf.squeeze(self.affinity_score(src_emb, pos_dst_emb), axis=-1)
    neg_score = tf.squeeze(self.affinity_score(src_emb, neg_dst_emb), axis=-1)
    return pos_score, neg_score