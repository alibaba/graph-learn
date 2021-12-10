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


class Dataset(object):
  """Dataset reformats the sampled batch from GSL query as `Data`
  consists of Numpy Arrays or generates `SubGraph`s using induce_func.

  Args:
    query (Dag): A GSL query, starts from .V()/.E(), ends up with .values().
    window (int, optional): Count of prefetched batches. Defaults to 5.
  """
  def __init__(self, query, window=10):
    self._dag = query
    self._ds = DagDataset(query, window)

    self._masks = OrderedDict()
    for alias in query.list_alias():
      dag_node = self._dag.get_node(alias)
      self._masks[alias] = self.get_mask(dag_node.decoder, 
                                         is_edge = isinstance(dag_node, TraverseEdgeDagNode),
                                         is_sparse = dag_node.sparse)

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
      assert len(sparse_masks) == 3 # offsets, indices, dense_shape
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
        values.extend([value.indices])
        values.extend([value.dense_shape])
      else:
        values.extend([None, None, None])
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

  def get_subgraphs(self, inducer):
    """Induce `SubGraph`/`HeteroSubGraph` using the `induce_func`.
    Args:
      inducer: A `SubGraphInducer` instance to generate SubGraph/HeteroSubGraph.
      as input and induce the `SubGraph`s.
    Return:
      a tuple of positive subgraphs and negative subgraphs.
    """
    try:
      values = self._ds.next()
      return inducer.induce_func(values)
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
      ints, floats, strings, labels, weights, ids, dst_ids, \
      offsets, indices, dense_shape = [pop(msk) for msk in sum(masks, [])]
      data_dict[alias] = Data(
        ids, ints, floats, strings, labels, weights, dst_ids=dst_ids,
        offsets=offsets, indices=indices, dense_shape=dense_shape)
    return data_dict

  def get_mask(self, node_decoder, is_edge=False, is_sparse=False):
    """The masks for features, ids and offsets.
    feat_masks: a list of boolean, each element indicates that data
    has int_attrs, float_attrs, string_attrs, lables, weights.
    id_masks: for Nodes is [True, False], for Edges is [True, True].
    sparse_masks: three boolean element list which indicates whether the object 
      is sparse.
    Args:
      node_decoder: The given node_decoder.

    Returns:
      (list, list, list): Masks for features, ids and offsets.
    """
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
    if is_edge:
      id_masks = [True, True]
    sparse_masks = [False, False, False]	
    if is_sparse:		
      sparse_masks = [True, True, True]

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
