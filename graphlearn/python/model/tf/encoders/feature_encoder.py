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

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

from graphlearn.python.model.base_encoder import BaseFeatureEncoder

EMB_PT_SIZE = 128 * 1024


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
  """

  def __init__(self,
               categorical_features_dict,
               total_feature_num,
               output_dim,
               use_input_bn=True,
               act=None,
               need_dense=True,
               ps_hosts=None,
               name=''):
    self._output_dim = output_dim
    self._categorical_features = \
      self._parse_feature_config(categorical_features_dict)
    self._feature_num = total_feature_num
    self._use_input_bn = use_input_bn
    self._need_dense = need_dense
    self._name=name

    self._emb_table = {}
    self._act = act

    emb_partitioner = None
    self.ps_num = 1

    if ps_hosts:
      ps_num = len(ps_hosts.split(","))
      emb_partitioner = \
        tf.min_max_variable_partitioner(max_partitions=ps_num,
                                        min_slice_size=EMB_PT_SIZE)

    with tf.variable_scope(self._name + 'feature_emb_table', reuse=tf.AUTO_REUSE):
      if self._categorical_features:
        for _, attr_name, max_num, emb_dim in self._categorical_features:
          self._emb_table[attr_name] = \
            tf.get_variable("emb_lookup_table_" + attr_name,
                            [max_num, emb_dim], partitioner=emb_partitioner)

  def _parse_feature_config(self, categorical_features_dict):
    categoricals = []
    if categorical_features_dict is not None \
      and categorical_features_dict: # list if not empty
      for k, v in categorical_features_dict.items():
        categoricals.append((int(k), v[0], v[1], v[2]))

    return categoricals

  def encode(self, input_attrs):
    """Encode input_attrs to embeddings.

    Args:
      input_attrs: A list in the format of [continuous_attrs, categorical_attrs]

    Returns:
      Embeddings.
    """

    continuous_attrs = input_attrs[0]
    categorical_attrs = input_attrs[1]

    to_concats_cate = []
    if self._categorical_features:
      for idx, attr_name, max_num, _ in self._categorical_features:
        attr = categorical_attrs[:, idx]
        attr = tf.string_to_hash_bucket_fast(
            attr,
            max_num,
            name=self._name + 'to_hash_bucket_%s' % (attr_name))

        to_concats_cate.append(
            tf.nn.embedding_lookup(
                self._emb_table[attr_name],
                attr,
                name=self._name + 'embedding_lookup_%s' % (attr_name)))

    to_concats_con = None
    continuous_feats_num = self._feature_num - len(self._categorical_features)
    if continuous_feats_num > 0:  # contains continuous features
      to_concats_con = tf.log(
          tf.reshape(continuous_attrs, [-1, continuous_feats_num]) + 2)

    with tf.variable_scope(self._name + 'attrs_encoding', reuse=tf.AUTO_REUSE):
      raw_emb_cate = None
      if to_concats_cate: # list is not empty
        raw_emb_cate = tf.concat(to_concats_cate, axis=-1, name="cate_concat")

      if to_concats_con is not None:
        raw_emb_con = to_concats_con
        if self._use_input_bn:
          raw_emb_con = \
            tf.layers.batch_normalization(raw_emb_con, training=True)

        if raw_emb_cate is not None:
          raw_emb = \
            tf.concat([raw_emb_cate, raw_emb_con],
                      axis=-1, name='con_cate_concat')
        else:
          raw_emb = raw_emb_con
      else:
        print('no continuous feature to emb')
        raw_emb = raw_emb_cate

      if self._need_dense:
        raw_emb = \
          tf.layers.dense(raw_emb, self._output_dim,
                          activation=self._act, name='dense')

    return raw_emb


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
