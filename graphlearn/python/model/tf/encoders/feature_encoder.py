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
"""Classes for encoding features to embeddings.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from graphlearn.python.model.base_encoder import BaseFeatureEncoder

EMB_PT_SIZE = 128 * 1024
DNN_PT_SIZE = 32 * 1024


class WideNDeepEncoder(BaseFeatureEncoder):
  """Encode categorical and continuous attributes(features) to embeddings.

  Wide and Deep encoder is proposed by Google researchers
  in the paper Wide & Deep Learning for Recommended Systems
  (https://arxiv.org/abs/1606.07792), the main idea is to
  use a deep neural network for continues attributes, and
  use a wide generalized linear model for categorical attributes,
  these two parts are then fused as the final embedding output.
  Check their paper for more details.

  Args:
    categorical_features_dict: A dict indicates discrete features, with the
      format {feature columun index : [name, max number, embedding dimension]}.
    total_feature_num: Total features number.
    output_dim: output dimension.
    use_input_bn: Whether to use batch normalization.
    act: Activation function.
    need_dense: Whether to use dense layer before output.
    ps_hosts: ps_hosts used in distributed TF training.
    name: User defined name.
    multivalent_cols_num: the number of multivalent columns,
      which is at the end of attributes list.
  """

  def __init__(self,
               categorical_features_dict,
               total_feature_num,
               output_dim,
               use_input_bn=True,
               act=None,
               need_dense=True,
               ps_hosts=None,
               name='',
               is_training=True,
               multivalent_cols_num=0):
    self._output_dim = output_dim
    self._feature_num = total_feature_num
    self._act = act
    self._use_input_bn = use_input_bn
    self._need_dense = need_dense
    self._name=name
    self._is_training = is_training

    self._categorical_features = \
      self._parse_feature_config(categorical_features_dict)
    if multivalent_cols_num > 0:
      self._multivalent_features = self._categorical_features[-multivalent_cols_num:]
      self._categorical_features = self._categorical_features[:-multivalent_cols_num]
    else:
      self._multivalent_features = []

    partitioner = None
    if ps_hosts:
      partitioner = \
        tf.min_max_variable_partitioner(max_partitions=len(ps_hosts.split(",")),
                                        min_slice_size=EMB_PT_SIZE)
    self._emb_table = {}
    self._offsets = {}
    with tf.variable_scope(self._name + 'feature_emb_table', reuse=tf.AUTO_REUSE):
      if self._categorical_features:
        total_max_num = 0
        self.emb_dim = 16
        for idx, attr_name, max_num, emb_dim in self._categorical_features:
          self.emb_dim = emb_dim
          self._offsets[idx] = tf.constant(total_max_num)
          total_max_num += max_num
        self._emb_table["coalesced_embed"] = \
          tf.get_variable("emb_lookup_table_",
                          [total_max_num, self.emb_dim],
                          partitioner=partitioner)
      if self._multivalent_features:
        for _, attr_name, max_num, emb_dim in self._multivalent_features:
          self._emb_table[attr_name] = \
            tf.get_variable("emb_lookup_sparse_table_" + attr_name,
                            [max_num, emb_dim],
                            partitioner=partitioner)

  def encode(self, input_attrs):
    """Encode input_attrs to embeddings.

    Args:
      input_attrs: A list in the format of [continuous_attrs, categorical_attrs]

    Returns:
      Embeddings.
    """
    continuous_attrs = input_attrs[0]
    categorical_attrs = input_attrs[1]
    to_concats_cate = None
    if self._categorical_features:
      coalesced_attrs = []
      for idx, attr_name, max_num, _ in self._categorical_features:
        attr = categorical_attrs[:, idx] + self._offsets[idx]
        coalesced_attrs.append(attr)
      with tf.device('/cpu:0'):
        attrs = tf.reshape(tf.stack(coalesced_attrs, axis=-1), [-1])
        to_concats_cate = tf.nn.embedding_lookup(self._emb_table["coalesced_embed"],
                                                 attrs,
                                                 name=self._name + 'embedding_lookup',
                                                 unique=True)
    if self._multivalent_features:
      for idx, attr_name, max_num, _ in self._multivalent_features:
        sparse_attr = tf.strings.split(categorical_attrs[:, idx], "|")
        ids = tf.string_to_hash_bucket_fast(sparse_attr.values, max_num,
                                            name=self._name + 'to_hash_bucket_%s' % (attr_name))
        sparse_ids = tf.SparseTensor(sparse_attr.indices, ids, sparse_attr.dense_shape)
        with tf.device('/cpu:0'):
          to_concats_cate.append(
            tf.nn.embedding_lookup_sparse(self._emb_table[attr_name],
                                          sp_ids=sparse_ids, sp_weights=None, combiner='mean',
                                          name=self._name + 'embedding_lookup_sparse_%s' % (attr_name)))

    with tf.variable_scope(self._name + 'attrs_encoding', reuse=tf.AUTO_REUSE):
      raw_emb_con = None
      raw_emb_cate = None
      continuous_feats_num = self._feature_num - len(self._categorical_features) - \
                             len(self._multivalent_features)
      if continuous_feats_num > 0:  # contains continuous features
        raw_emb_con = tf.reshape(continuous_attrs, [-1, continuous_feats_num])
      if to_concats_cate is not None:
        raw_emb_cate = tf.reshape(to_concats_cate, [-1, len(self._categorical_features) * self.emb_dim])

      if raw_emb_con is not None:
        if self._use_input_bn:
          raw_emb_con = tf.layers.batch_normalization(raw_emb_con, training=self._is_training)
        raw_emb = raw_emb_con
        if raw_emb_cate is not None:
          raw_emb = tf.concat([raw_emb_cate, raw_emb], axis=-1, name='con_cate_concat')
      else:
        print('no continuous feature to emb')
        raw_emb = raw_emb_cate

      if self._need_dense:
        raw_emb = tf.layers.dense(raw_emb, self._output_dim, activation=self._act, name='dense')
      return raw_emb

  def _parse_feature_config(self, categorical_features_dict):
    categoricals = []
    if categorical_features_dict is not None \
      and categorical_features_dict: # list if not empty
      for k, v in categorical_features_dict.items():
        categoricals.append((int(k), v[0], v[1], v[2]))
    return categoricals


class IdentityEncoder(BaseFeatureEncoder):
  """A identity encoder which returns continuous part of input_attrs.
  """
  def __init__(self):
    pass

  def encode(self, input_attrs):
    """Return continuous part of input_attrs

    Args:
      input_attrs: A list in the format of [continuous_attrs, categorical_attrs]

    Returns:
      continuous_attrs
    """
    continuous_attrs, _ = input_attrs
    return continuous_attrs
