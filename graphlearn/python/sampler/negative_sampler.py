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

import numpy as np
from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python.utils import strategy2op
import graphlearn.python.errors as errors


class NegativeSampler(object):
  """ Negative sampling from a graph.
  """

  def __init__(self,
               graph,
               object_type,
               expand_factor,
               strategy="random"):
    """ Create a Base NegativeSampler instance.

    Args:
      graph (`Graph` object): The graph which sample from.
      object_type (string): Sample negative nodes of the source node with
        specified edge_type or node_type.
      expand_factor: An integer, how many negative ids will be sampled
        for each given id.
      strategy (string): "random", "in_degree", "node_weight" are supported.
    """
    self._graph = graph
    self._object_type = object_type
    self._expand_factor = expand_factor
    self._strategy = strategy
    self._client = self._graph.get_client()

    if object_type in self._graph.get_node_decoders():
      self._dst_type = object_type
    elif object_type in self._graph.get_edge_decoders():
      topology = self._graph.get_topology()
      self._dst_type = topology.get_dst_type(object_type)
    else:
      raise ValueError("node or edge type {} is not in the graph"
                       .format(object_type))

    self._check()

  def _check(self):
    pass

  def get(self, ids):
    """ Get batched negative samples.

    Args:
      ids (numpy.array): A 1d numpy array of whose negative dst nodes
        will be sampled.

    Return:
      A `Nodes` object, shape=[ids.shape, `expand_factor`].
    """
    ids = np.array(ids).flatten()

    req = self._make_req(ids)
    res = pywrap.new_sampling_response()
    status = self._client.sample_neighbor(req, res)
    if status.ok():
      nbrs = pywrap.get_sampling_node_ids(res)
      neg_nbrs = self._graph.get_nodes(
          self._dst_type, nbrs, shape=(ids.shape[0], self._expand_factor))

    pywrap.del_op_response(res)
    pywrap.del_op_request(req)
    errors.raise_exception_on_not_ok_status(status)
    return neg_nbrs

  def _make_req(self, ids):
    sampler = strategy2op(self._strategy, "NegativeSampler")
    req = pywrap.new_sampling_request(
      self._object_type, sampler, self._expand_factor, 0)
    pywrap.set_sampling_request(req, ids)
    return req


class RandomNegativeSampler(NegativeSampler):
  def _check(self):
    assert self._object_type in self._graph.get_edge_decoders(), \
           "{} is not type of edge.".format(self._object_type)


class InDegreeNegativeSampler(NegativeSampler):
  def _check(self):
    assert self._object_type in self._graph.get_edge_decoders(), \
           "{} is not type of edge.".format(self._object_type)


class NodeWeightNegativeSampler(NegativeSampler):
  def _check(self):
    assert self._object_type in self._graph.get_node_decoders(), \
           "{} is not type of node.".format(self._object_type)
    assert self._graph.get_node_decoder(self._object_type).weighted, \
           "node {} is not WEIGHTED.".format(self._object_type)


class ConditionalNegativeSampler(NegativeSampler):
  """ Negative sampling undere specified attribute conditions.
  The attributes of the negative sampled nodes need to be the same
  as the atributes of dst nodse in the postive sample.
  in given positive sample.
  """

  def __init__(self,
               graph,
               object_type,
               expand_factor,
               strategy,
               **kwargs):
    """ Create a Base NegativeSampler instance.

    Args:
      batch_share: Whether sampled negative samples are shared by this batch.
      unique: Whether sampled negtive samples are unique.
      int_cols: int columns as condition.
      int_props: proportions of int columns.
      float_cols: float columns as condition.
      float_props: proportions of float columns.
      str_cols: string columns as condition.
      str_props: proportions of string columns.
    """
    super(ConditionalNegativeSampler, self).__init__(graph,
                                                     object_type,
                                                     expand_factor,
                                                     strategy)
    self._batch_share = kwargs.get("batch_share", False)
    self._unique = kwargs.get("unique", False)
    self._i_cols = kwargs.get("int_cols", [])
    self._i_props = kwargs.get("int_props", [])
    self._f_cols = kwargs.get("float_cols", [])
    self._f_props = kwargs.get("float_props", [])
    self._s_cols = kwargs.get("str_cols", [])
    self._s_props = kwargs.get("str_props", [])
    self._condition_check()

  def _condition_check(self):
    def check_cols_props_len_eq(cols, props):
      if not ((cols is None) == (props is None)):
        raise ValueError("Condition columns and props must be None \
            or not None at the same time.")
      elif cols is not None and len(cols) != len(props):
        raise ValueError("Condition columns and props must be the same size.")

    def check_attr_column_range(cols, col_range):
      if cols is not None:
        for i in cols:
          if not i < col_range:
            raise ValueError("Condition columns index out of range.")

    check_cols_props_len_eq(self._i_cols, self._i_props)
    check_cols_props_len_eq(self._f_cols, self._f_props)
    check_cols_props_len_eq(self._s_cols, self._s_props)

    attr_decoder = self._graph.get_node_decoder(self._dst_type)
    check_attr_column_range(self._i_cols, attr_decoder.int_attr_num)
    check_attr_column_range(self._f_cols, attr_decoder.float_attr_num)
    check_attr_column_range(self._s_cols, attr_decoder.string_attr_num)

    prob_sum = np.sum(self._i_props) if self._i_props is not None else 0
    prob_sum += np.sum(self._f_props) if self._f_props is not None else 0
    prob_sum += np.sum(self._s_props) if self._s_props is not None else 0
    if prob_sum > 1:
      raise ValueError("Condition props sum is greater than 1.")

  def get(self, src_ids, dst_ids):
    """ Get batched negative samples.

    Args:
      src_ids (numpy.array): A 1d numpy array of whose negative dst nodes
        will be sampled.
      dst_ids (numpy.array): A 1d numpy array of positive dst nodes.

    Return:
      A `Nodes` object, shape=[ids.shape, `expand_factor`].
    """
    src_ids = np.array(src_ids).flatten()
    dst_ids = np.array(dst_ids).flatten()

    req = self._make_req(src_ids, dst_ids)
    res = pywrap.new_sampling_response()
    status = self._client.cond_neg_sample(req, res)
    if status.ok():
      nbrs = pywrap.get_sampling_node_ids(res)
      neg_nbrs = self._graph.get_nodes(
          self._dst_type, nbrs, shape=(dst_ids.shape[0], self._expand_factor))

    pywrap.del_op_response(res)
    pywrap.del_op_request(req)
    errors.raise_exception_on_not_ok_status(status)
    return neg_nbrs

  def _make_req(self, src_ids, dst_ids):
    req = pywrap.new_conditional_sampling_request(
        self._object_type,
        self._strategy,
        self._expand_factor,
        self._dst_type,
        self._batch_share,
        self._unique)
    pywrap.set_conditional_sampling_request_ids(
        req, src_ids, dst_ids)
    pywrap.set_conditional_sampling_request_cols(
        req,
        self._i_cols, self._i_props,
        self._f_cols, self._f_props,
        self._s_cols, self._s_props)
    return req
