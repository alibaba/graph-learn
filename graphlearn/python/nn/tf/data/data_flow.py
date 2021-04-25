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

import numpy as np
from collections import OrderedDict
import tensorflow as tf

from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python.errors import OutOfRangeError
from graphlearn.python.gsl.dag_dataset import Dataset
from graphlearn.python.gsl.dag_node import TraverseEdgeDagNode
from graphlearn.python.nn.tf.data.entity import Vertex
from graphlearn.python.nn.tf.data.ego_graph import EgoGraph


class DataFlow(object):
  def __init__(self, query, window=5):
    self._dag = query
    self._ds = Dataset(query, window)
    self._handler_dict = OrderedDict()
    for alias in query.list_alias():
      self._handler_dict[alias] = self._get_types_and_shapes(alias)

    self._iterator = None

    self._entities = self._build()

  @property
  def iterator(self):
    return self._iterator

  def _get_types_and_shapes(self, alias):
    node = self._dag.get_node(alias)
    node_spec = node.spec

    feat_shapes = []
    feats = ('int_attr_num', 'float_attr_num', 'string_attr_num',
             'labeled', 'weighted')
    feat_types = np.array([tf.int64, tf.float32, tf.string,
                           tf.int32, tf.float32])
    feat_masks = [False] * len(feat_types)

    for idx, feat in enumerate(feats):
      feat_spec = getattr(node_spec, feat)
      if isinstance(feat_spec, bool) and feat_spec: # For labels/weights
        feat_masks[idx] = True
        feat_shapes.append(tf.TensorShape([None]))
      elif feat_spec > 0: # For attrs
        feat_masks[idx] = True
        feat_shapes.append(tf.TensorShape([feat_spec, None]))

    id_types = [tf.int64]
    id_shapes = [tf.TensorShape([None])]
    id_masks = [True, False]  # Default: (ids, None), else: (src_ids, dst_ids)
    if isinstance(node, TraverseEdgeDagNode):
      id_types.append(tf.int64)  # Edges: src_ids, dst_ids
      id_shapes.append(tf.TensorShape([None]))
      id_masks = [True, True]

    degree_types = [{}, {}]  # in_degrees, out_degrees
    degree_shapes = [{}, {}]
    for dg_node in node.get_degree_nodes():
      choice = int(dg_node.node_from == pywrap.NodeFrom.EDGE_DST)
      degree_types[choice][dg_node.edge_type] = tf.int32
      degree_shapes[choice][dg_node.edge_type] = tf.TensorShape([None])
    degree_masks = [len(degree_types[0]) > 0, len(degree_types[1]) > 0]
    degree_types = [dtp for dtp in degree_types if len(dtp) > 0]
    degree_shapes = [dsp for dsp in degree_shapes if len(dsp) > 0]

    return list(feat_types[feat_masks]) + id_types + degree_types, \
          feat_shapes + id_shapes + degree_shapes, \
          feat_masks, id_masks, degree_masks

  def _reorganize_features(self,
                           int_attrs, float_attrs, string_attrs,
                           labels, weights):
    """ Reformats the SEED Nodes/Edges traversed from graph store, or the
    NEIGHBOR Nodes/Edges sampled from graph store.
    For SEED:
      Reorganize shape of attributes:
        [batch_size, attr_num]
          --reshape--> [batch_size, attr_num]
          -transpose-> [attr_num, batch_size]
      Reorganize shape of weights/labels:
        [batch_size, ]
          --flatten--> [batch_size, ]
    For NEIGHBOR:
      Reorganize shape of attributes:
        [batch_size, nbr_count, attr_num]
          --reshape--> [batch_size * nbr_count, attr_num]
          -transpose-> [attr_num, batch_size * nbr_count]
      Reorganize shape of weights/labels:
        [batch_size, nbr_count]
          --flatten--> [batch_size * nbr_count, ]
    """
    def reshape(feat):
      if feat is not None:
        return np.reshape(feat, (-1, feat.shape[-1]))
      return feat

    def transpose(feat):
      if feat is not None:
        return np.transpose(feat)
      return feat
    
    def flatten(feat):
      if feat is not None:
        return feat.flatten()
      return feat

    int_attrs = transpose(reshape(int_attrs))
    float_attrs = transpose(reshape(float_attrs))
    string_attrs = transpose(reshape(string_attrs))
    lables = flatten(labels)
    weights = flatten(weights)
    return [int_attrs, float_attrs, string_attrs, lables, weights]


  def _generator(self):
    def parse_value(node, feat_masks, id_masks, degree_masks):
      assert len(feat_masks) == 5  # i_attrs, f_attrs, s_attrs, lables, weights
      assert len(id_masks) == 2  # src_ids, dst_ids
      assert len(degree_masks) == 2  # out_degrees, in_degrees

      values = self._reorganize_features(
        node.int_attrs, node.float_attrs, node.string_attrs,
        node.labels, node.weights)
      if id_masks[-1]:  # dst_ids existed
        values.extend([node.src_ids.flatten(), node.dst_ids.flatten()])
      else:
        values.extend([node.ids.flatten(), None])
      values.extend([node.out_degrees, node.in_degrees])
      return list(np.array(values)[feat_masks + id_masks + degree_masks])

    while True:
      try:
        values = self._ds.next()
        res = []
        for alias, handler in self._handler_dict.items():
          node = values[alias]
          _, _, feat_masks, id_masks, degree_masks = handler
          res.extend(parse_value(node, feat_masks, id_masks, degree_masks))
        yield tuple(res)
      except OutOfRangeError:
        break

  def _build(self):
    # output_types and oytput_shapes are ordered by
    # alias order in OrderedDict(handler_dict).
    output_types = []
    output_shapes = []
    for types, shapes, _, _, _ in self._handler_dict.values():
      output_types.extend(types)
      output_shapes.extend(shapes)
    dataset = tf.data.Dataset.from_generator(self._generator,
                                             tuple(output_types),
                                             tuple(output_shapes))
    self._iterator = dataset.make_initializable_iterator()
    value = self._iterator.get_next()
    return self._build_entity(value)

  def _build_entity(self, value):
    entities = {}
    cursor = [-1]

    def pop(mask):
      if mask:
        cursor[0] += 1
        return value[cursor[0]]
      return None

    for alias, handler in self._handler_dict.items():
      _, _, feat_masks, id_masks, degree_masks = handler
      ints, floats, strings, labels, weights, ids, _, out_degrees, in_degrees = \
        [pop(msk) for msk in feat_masks + id_masks + degree_masks]
      # TODO(wenting.swt): To support edges.
      entities[alias] = Vertex(
        ids, ints, floats, strings, labels, weights, out_degrees, in_degrees)
    return entities

  def get(self, alias):
    """Get the entity of given alias.

    Args:
      alias (str): alias in the GSL query.
    """
    return self._entities[alias]

  def get_ego_graph(self, source, neighbors=None):
    """ Reformat the tensors as EgoGraph.
    Args:
      source(str): alias of centric vertices.
      neighbors(list of str): alias of neighbors at each hop.
        Default `None`: automatically generating the positive neighbors for
        centric vertices. It requires that each hop only has one postive
        downstream in GSL.
        Given list of string: the alias of each hop in GSL. The list must
        follow the order of traverse in GSL, and each one should be the postive
        or negative downstream for the front.
    """
    def _get_node_type(alias):
      return self._dag.get_node(alias).type

    def _get_upstream_edge(alias):
      return self._dag.get_node(alias)._path

    def _get_feat_spec(alias):
      node = self._dag.get_node(alias)
      decoder = node.spec
      return decoder.feature_spec

    source_node = self._dag.get_node(source)
    nbrs = []
    hops = []

    if neighbors:
      # Use specified neighbors to construct EgoGraph.
      if not isinstance(neighbors, list):
        raise ValueError("`neighbors` should be a list of alias")
      pre = source_node
      for nbr in neighbors:
        nbr_node = self._dag.get_node(nbr)
        if not nbr_node in pre.pos_downstreams + pre.neg_downstreams:
          raise ValueError("{} is not the downstream of {}.".format(
            nbr_node.get_alias(), pre.get_alias()))
        nbrs.append(nbr_node.get_alias())
        hops.append(nbr_node.shape[-1])
        pre = nbr_node

    else:
      # Use default receptive neighbors to construct EgoGraph.
      pre = source_node
      recepts = source_node.pos_downstreams
      while recepts:
        if len(recepts) > 1:
          raise ValueError("Can't automatically find neighbors for {},"
                           " which has multiple downstreams. You should"
                           " assign specific neighbors for {}."
                           .format(pre.get_alias(), source))
        pre = recepts[0]
        nbrs.append(pre.get_alias())
        hops.append(pre.shape[-1])
        recepts = pre.pos_downstreams

    # Only keep the degrees with the traversing path in the specific EgoGraph.
    all_alias = [source] + nbrs
    for idx  in range(len(all_alias) - 1):
      etype = _get_upstream_edge(all_alias[idx + 1])
      entity = self._entities[all_alias[idx]]
      entity.register_handler("out_edge_type", etype)

    return EgoGraph(self._entities[source],
                    [self._entities[nbr] for nbr in nbrs],
                    [(_get_node_type(v), _get_feat_spec(v)) for v in [source] + nbrs],
                    hops)

  def get_sub_graph(self, alias):
    """
    Reformat the tensors as SubGraph.
    """
    pass
