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
from graphlearn.python.data.state import DagState
from graphlearn.python.errors import OutOfRangeError, \
  raise_exception_on_not_ok_status

global_dag_state = DagState()

class Dataset(object):
  def __init__(self, dag, window=10):
    assert dag.is_ready(), \
      "Query should start with E()/V() and end with values()."
    assert isinstance(window, int) and 0 < window < 128, \
      "Dataset window should be in range of (0, 128)."

    self._dag = dag
    self._dag_id = dag.name
    self._cur_res = None

    graph = dag.graph
    client = graph.get_client()
    status = client.run_dag(self._dag.dag_def)
    raise_exception_on_not_ok_status(status)

    pywrap.set_dataset_capacity(window)
    self._dag_dataset = pywrap.Dataset(client, self._dag_id)
    graph.add_dataset(self)

  def next(self):
    # Delete the response of last round.
    if self._cur_res:
      pywrap.del_get_dag_value_response(self._cur_res)

    # New response.
    state = global_dag_state.get(self._dag.name)

    # Fill response.
    res = self._dag_dataset.next(state)

    if res and res.valid():
      dag_values = DagValues(self._dag, res)
      result = dag_values.process()
      self._cur_res = res
      return result
    else:
      global_dag_state.inc(self._dag.name)
      if res:
        pywrap.del_get_dag_value_response(res)
      self._cur_res = None
      raise OutOfRangeError("OutOfRange")

  def close(self):
    self._dag_dataset.close()


class DagValues(object):
  def __init__(self, dag, res):
    # The values already got from response.
    self._cache = {}
    self._dag = dag
    # graph is needed to construct Nodes/Edges
    self._graph = dag.graph
    # res is where to get data from.
    self._res = res

  def process(self):
    return self._dag.value_func(self)

  def __getitem__(self, alias):
    # Get the DagNode of alias.
    node = self._dag.get_node(alias)
    if not node:
      raise ValueError("Alias {} is not existed in the query.".format(alias))

    # If Nodes/Edges of the alias already got from response,
    # just get them from python values, instead of converting from C++.
    if alias in self._cache.keys():
      return self._cache[alias]

    # Cache Nodes/Edges.
    res = node.feed_values(self._res)
    self._cache[alias] = res

    # Add the attributes for the Nodes/Edges.
    lookup_node = node.get_lookup_node()
    res.int_attrs = pywrap.get_dag_value(self._res, lookup_node.nid, "ia")
    res.float_attrs = pywrap.get_dag_value(self._res, lookup_node.nid, "fa")
    res.string_attrs = pywrap.get_dag_value(self._res, lookup_node.nid, "sa")
    res.weights = pywrap.get_dag_value(self._res, lookup_node.nid, "wei")
    res.labels = pywrap.get_dag_value(self._res, lookup_node.nid, "lb")

    # Add degrees for the Nodes.
    for dg_node in node.get_degree_nodes():
      if dg_node.node_from == pywrap.NodeFrom.EDGE_SRC:
        res.add_out_degrees(dg_node.edge_type,
                            pywrap.get_dag_value(self._res, dg_node.nid, "dg"))
      if dg_node.node_from == pywrap.NodeFrom.EDGE_DST or \
          dg_node.edge_type in self._graph.undirected_edges:
        res.add_in_degrees(dg_node.edge_type,
                           pywrap.get_dag_value(self._res, dg_node.nid, "dg"))
    return res
