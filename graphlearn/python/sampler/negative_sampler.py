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
""" Negative samplers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python.errors import raise_exception_on_not_ok_status
from graphlearn.python.utils import strategy2op


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
      strategy (string): "random", "in_degree" are supported.
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
      raise ValueError("node or edeg tyep {} is not in the graph"
                       .format(object_type))

    self._check()

  def _check(self):
    pass

  def get(self, ids):
    """ Get batched samples.

    Args:
      ids (numpy.array): A 1d numpy array of whose negative dst nodes
        will be sampled.

    Return:
      A `Nodes` object, shape=[ids.shape, `expand_factor`].
    """
    if not isinstance(ids, np.ndarray):
      raise ValueError("ids must be a numpy array, got {}."
                       .format(type(ids)))
    ids = ids.flatten()

    req = self._make_req(ids)
    res = pywrap.new_sampling_response()
    status = self._client.sample_neighbor(req, res)
    if status.ok():
      nbrs = pywrap.get_sampling_node_ids(res)
      neg_nbrs = self._graph.get_nodes(
          self._dst_type, nbrs, shape=(ids.shape[0], self._expand_factor))

    pywrap.del_op_response(res)
    pywrap.del_op_request(req)
    raise_exception_on_not_ok_status(status)
    return neg_nbrs

  def _make_req(self, ids):
    sampler = strategy2op(self._strategy, "NegativeSampler")
    req = pywrap.new_sampling_request(self._object_type, sampler, self._expand_factor)
    pywrap.set_sampling_request(req, ids)
    return req


class RandomNegativeSampler(NegativeSampler):
  pass


class InDegreeNegativeSampler(NegativeSampler):
  pass


class NodeWeightNegativeSampler(NegativeSampler):
  def _check(self):
    assert self._object_type in self._graph.get_node_decoders(), \
           "{} is not type of node.".format(self._object_type)
    assert self._graph.get_node_decoder(self._object_type).weighted, \
           "node {} is not WEIGHTED.".format(self._object_type)
