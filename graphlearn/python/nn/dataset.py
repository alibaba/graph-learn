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
from enum import Enum

from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python.errors import OutOfRangeError
from graphlearn.python.gsl.dag_dataset import Dataset as DagDataset
from graphlearn.python.gsl.dag_node import TraverseEdgeDagNode
from graphlearn.python.nn.data import Data
from graphlearn.python.nn.utils.induce_graph_with_edge import \
  induce_graph_with_edge

class SubKeys(Enum):
  POS_SRC = 0
  NGE_SRC = 1 # not support yet.
  POS_DST = 2
  NEG_DST = 3
  POS_EDGE = 4 # not support yet.
  NEG_EDGE = 5 # not support yet.


class Dataset(object):
  """Dataset reformats the sampled batch from GSL query as `Data`
  consists of Numpy Arrays or generates `SubGraph`s using induce_func.

  Args:
    query (Dag): A GSL query, starts from .V()/.E(), ends up with .values().
    window (int, optional): Count of prefetched batches. Defaults to 5.
  """
  def __init__(self, query, window=5):
    self._dag = query
    self._ds = DagDataset(query, window)

    self._masks = OrderedDict()
    for alias in query.list_alias():
      self._masks[alias] = self._get_mask(alias)

  def __iter__(self):
    def iterator():
      while True:
        try:
          yield self.get_data_dict()
        except OutOfRangeError:
          break
    return iterator()

  @property
  def iterator(self):
    """Get an iterator, which generates `Data` objects.
    """
    return self.__iter__()

  @property
  def masks(self):
    """The masks of features, ids, degrees and sparse_offsets for 
    each `Data` object.
    """
    return self._masks

  def get_flatten_values(self):
    """Get and reformat the raw flatten numpy values list from query.
    """
    def parse_value(value, masks):
      assert len(masks) == 3
      feat_masks, id_masks, sparse_masks = masks
      assert len(feat_masks) == 5  # i_attrs, f_attrs, s_attrs, lables, weights
      assert len(id_masks) == 2  # src_ids, dst_ids
      assert len(sparse_masks) == 1 # offsets
      # Features
      values = self._reformat_features(
        value.int_attrs, value.float_attrs, value.string_attrs,
        value.labels, value.weights)
      # Ids
      if id_masks[-1]:  # dst_ids existed
        values.extend([value.src_ids.flatten(), value.dst_ids.flatten()])
      else:
        values.extend([value.ids.flatten(), None])
      # Offsets for Sparse Neighbors
      if sparse_masks[-1]:
        values.extend([value.offsets])
      else:
        values.extend([None])
      return list(np.array(values)[feat_masks + id_masks + sparse_masks])

    try:
      values = self._ds.next()
      res = []
      for alias, masks in self._masks.items():
        value = values[alias]
        res.extend(parse_value(value, masks))
      return res
    except OutOfRangeError:
      raise OutOfRangeError("out of range.")

  def get_data_dict(self):
    """ Get dict of `Data` in numpy format from flatten values.
    """
    return self.build_data_dict(self.get_flatten_values())

  def get_subgraphs(self, induce_func=induce_graph_with_edge):
    """Induce `SubGraph`s using the `induce_func`.
    Args:
      induce_func: an induce `SubGraph` function which takes the query result
      as input and induce the `SubGraph`s.
    Return:
      a tuple of positive subgraphs and negative subgraphs.
    """
    #TODO(baole): supports generator with node.
    try:
      values = self._ds.next()
      pos_src = values[SubKeys.POS_SRC]
      # TODO(baole): support multi-hops
      src_nbrs = values[self._dag.get_node(SubKeys.POS_SRC).\
        pos_downstreams[0].get_alias()]
      if SubKeys.POS_DST in self._dag.list_alias():
        pos_dst = values[SubKeys.POS_DST]
        dst_nbrs = values[self._dag.get_node(SubKeys.POS_DST).\
          pos_downstreams[0].get_alias()]
      else: # fake a src-src edge.
        pos_dst, dst_nbrs = pos_src, src_nbrs
      subgraphs = induce_func(pos_src, pos_dst, src_nbrs, dst_nbrs)
      # negative samples.
      neg_subgraphs = None
      if SubKeys.NEG_DST in self._dag.list_alias():
        neg_dst = values[SubKeys.NEG_DST]
        neg_dst_nbrs = values[self._dag.get_node(SubKeys.NEG_DST).\
          pos_downstreams[0].get_alias()]
        neg_subgraphs = induce_func(pos_src, neg_dst, 
          src_nbrs, neg_dst_nbrs)
      return subgraphs, neg_subgraphs
    except OutOfRangeError:
      raise OutOfRangeError("out of range.")


  def build_data_dict(self, flatten_values):
    """Build the dict of Data from flatten value lists.

    Returns:
        dict: key is alias, value is `Data`.
    """
    data_dict = {}
    cursor = [-1]

    def pop(mask):
      if mask:
        cursor[0] += 1
        return flatten_values[cursor[0]]
      return None

    for alias, masks in self._masks.items():
      ints, floats, strings, labels, weights, ids, dst_ids, offsets = \
        [pop(msk) for msk in sum(masks, [])]
      data_dict[alias] = Data(
        ids, ints, floats, strings, labels, weights, 
        offsets=offsets, dst_ids=dst_ids)
    return data_dict

  def _get_mask(self, alias):
    """The masks for features, ids and offsets.
    feat_masks: a list of boolean, each element indicates that data
    has int_attrs, float_attrs, string_attrs, lables, weights.
    id_masks: for Nodes is [True, False], for Edges is [True, True].
    sparse_masks: one boolean element list that indicates whether the object 
    is sparse.
    Args:
      alias (str): alias in GSL query.

    Returns:
      (list, list, list): Masks for features, ids and offsets.
    """
    node = self._dag.get_node(alias)
    node_decoder = node.decoder

    feats = ('int_attr_num', 'float_attr_num', 'string_attr_num',
             'labeled', 'weighted')
    feat_masks = [False] * len(feats)
    for idx, feat in enumerate(feats):
      feat_spec = getattr(node_decoder, feat)
      if isinstance(feat_spec, bool) and feat_spec: # For labels/weights
        feat_masks[idx] = True
      elif feat_spec > 0: # For attrs
        feat_masks[idx] = True

    id_masks = [True, False]  # Default: (ids, None), else: (src_ids, dst_ids)
    if isinstance(node, TraverseEdgeDagNode):
      id_masks = [True, True]
    
    sparse_masks = [False]
    if node.sparse:
      sparse_masks[-1] = True

    return feat_masks, id_masks, sparse_masks

  def _reformat_features(self,
                           int_attrs,
                           float_attrs,
                           string_attrs,
                           labels,
                           weights):
    """ Reformats the SEED Nodes/Edges traversed from graph store, or the
    NEIGHBOR Nodes/Edges sampled from graph store.
    For SEED:
      Reorganize shape of attributes:
        [batch_size, attr_num]
          --reshape--> [batch_size, attr_num]
      Reorganize shape of weights/labels:
        [batch_size, ]
          --flatten--> [batch_size, ]
    For NEIGHBOR:
      Reorganize shape of attributes:
        [batch_size, nbr_count, attr_num]
          --reshape--> [batch_size * nbr_count, attr_num]
      Reorganize shape of weights/labels:
        [batch_size, nbr_count]
          --flatten--> [batch_size * nbr_count, ]
    """
    def reshape(feat):
      if feat is not None:
        return np.reshape(feat, (-1, feat.shape[-1]))
      return feat

    def flatten(feat):
      if feat is not None:
        return feat.flatten()
      return feat

    int_attrs = reshape(int_attrs)
    float_attrs = reshape(float_attrs)
    string_attrs = reshape(string_attrs)
    lables = flatten(labels)
    weights = flatten(weights)
    return [int_attrs, float_attrs, string_attrs, lables, weights]
