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
from graphlearn.python.data.values import SubGraph
from graphlearn.python.errors import OutOfRangeError, \
  raise_exception_on_not_ok_status

global_dag_state = DagState()

class Dataset(object):
  def __init__(self, dag, window=10, keep_alive_rounds=1, drop_last=False):
    assert dag.is_ready(), \
      "Query should start with E()/V() and end with values()."
    assert isinstance(window, int) and 0 < window < 128, \
      "Dataset window should be in range of (0, 128)."

    self._dag = dag
    self._dag_id = dag.name
    self.keep_alive_rounds = keep_alive_rounds
    self.drop_last = drop_last
    self._last_responses = []

    self._graph = dag.graph
    self._client = self._graph.get_client()
    pywrap.set_dataset_capacity(window)

    self._dag_datasets = []
    # initialize the dataset for the all possible clients, note that the dag_def may
    # need to be copied to avoid been moved
    for index in range(len(self._client)):
      status = self._client.run_dag(self._dag.dag_def, index != len(self._client) - 1)
      raise_exception_on_not_ok_status(status)
      self._dag_datasets.append(pywrap.Dataset(self._client.current_client, self._dag_id))
      # move to the next client
      self._client.connect_to_next_server()

    # starting from the dataset from the first client
    self._current_index = 0
    # associate to graph to delete it when the graph been deleted
    self._graph.add_dataset(self)

  def del_last_responses(self):
    [pywrap.del_get_dag_value_response(res) for res in self._last_responses]
    self._last_responses = []

  def next(self):
    # Delete the response of last `keep_alive_rounds`` round.
    if len(self._last_responses) == self.keep_alive_rounds:
      self.del_last_responses()

    # New response.
    state = global_dag_state.get(self._dag.name)

    # Fill response.
    res = self._dag_datasets[self._current_index].next(state)

    if res and res.valid():
      dag_values = DagValues(self._dag, res)
      result = dag_values.process()
      self._last_responses.append(res)
      count = dag_values[self._dag.list_alias()[0]].shape[0]
      if self.drop_last and (count < self._dag.batch_size):
        return self.next()
      return result
    else:
      global_dag_state.inc(self._dag.name)
      if res:
        self._last_responses.append(res)
      self.del_last_responses()
      raise OutOfRangeError("OutOfRange")

  def close(self):
    for dataset in self._dag_datasets:
      if dataset is not None:
        dataset.close()


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
    if isinstance(res, SubGraph):
      data = res.nodes
    else:
      data = res
    if lookup_node is not None:
      data.int_attrs = pywrap.get_dag_value(self._res, lookup_node.nid, "ia")
      data.float_attrs= pywrap.get_dag_value(self._res, lookup_node.nid, "fa")
      data.string_attrs = pywrap.get_dag_value(self._res, lookup_node.nid, "sa")
      data.weights = pywrap.get_dag_value(self._res, lookup_node.nid, "wei")
      data.labels = pywrap.get_dag_value(self._res, lookup_node.nid, "lb")
      data.timestamps = pywrap.get_dag_value(self._res, lookup_node.nid, "ts")

    # Add degrees for the Nodes.
    for dg_node in node.get_degree_nodes():
      if dg_node.node_from == pywrap.NodeFrom.EDGE_SRC:
        data.add_out_degrees(dg_node.edge_type,
                             pywrap.get_dag_value(self._res, dg_node.nid, "dg"))
      if dg_node.node_from == pywrap.NodeFrom.EDGE_DST or \
          dg_node.edge_type in self._graph.undirected_edges:
        data.add_in_degrees(dg_node.edge_type,
                           pywrap.get_dag_value(self._res, dg_node.nid, "dg"))
    return res
