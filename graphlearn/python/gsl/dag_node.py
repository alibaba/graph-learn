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

import sys
# For python version 3.x, rename unicode as str.
if (int(sys.version_info[0]) == 3):
  unicode = str
import uuid
import numpy as np
import warnings

from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python.data.values import Nodes
from graphlearn.python.gsl.dag_edge import get_dag_edge, get_eid
from graphlearn.python.utils import strategy2op


class DagNode(object):
  def __init__(self, dag, op_name="", params={}):
    self._dag = dag
    self._op_name = op_name

    # Lazy init.
    self._alias = None
    self._params = {}  # keep params for the node
    self._output_field = None
    self._shape = None
    self._sparse = False
    self._path = None  # Upstream edge type
    self._type = None  # Edge type or node type of current DagNode
    self._strategy = "random"

    # Add by GSL.
    # All the DagEdges expect link to sink node are mapped to DagEdgeDef.
    self._out_edges = []  # downstream DagEdges
    self._in_edges = []  # upstream DagEdges

    # Indicates setting which field to src_input of the downstream edges
    self._graph = self._dag.graph
    self._decoder = None

    self._lookup_node = None  # Each traverse node has a LookupDagNode
    self._degree_nodes = []  # Traverse node may have several DegreeDagNodes

    # Init until Dag is ready.
    self._node_def = None
    self._nid = -1

    self._params.update(params)
    # Downstream Traverse DagNodes
    self._pos_downstreams = []
    # Downstream Negative Sampled Traverse DagNodes
    self._neg_downstreams = []

  def _add_degree_node(self, edge_type, node_from):
    edge = self._new_edge(dst_input=pywrap.kNodeIds)
    # Add an edge from upstream node to degree node.
    self._add_out_edge(edge)
    node =  DegreeDagNode(
      "GetDegree", self, [edge],
      {pywrap.kEdgeType: edge_type,
       pywrap.kNodeFrom: int(node_from),
       pywrap.kPartitionKey: pywrap.kNodeIds})
    self._degree_nodes.append(node)

  @property
  def decoder(self):
    return self._decoder

  @property
  def nid(self):
    return self._nid

  @property
  def op_name(self):
    return self._op_name

  @property
  def in_edges(self):
    return self._in_edges

  @property
  def out_edges(self):
    return self._out_edges

  @property
  def pos_downstreams(self):
    return self._pos_downstreams

  @property
  def neg_downstreams(self):
    return self._neg_downstreams

  def set_path(self, path, node_from):
    assert isinstance(path, str)
    assert isinstance(node_from, pywrap.NodeFrom)
    topo = self._graph.get_topology()
    path_to_type = {pywrap.NodeFrom.NODE: lambda x: x,
                    pywrap.NodeFrom.EDGE_SRC: topo.get_src_type,
                    pywrap.NodeFrom.EDGE_DST: topo.get_dst_type}
    self._path = path
    self._type = path_to_type.get(node_from)(path)
    if self._type in self._graph.get_node_decoders().keys():
      self._decoder = self._graph.get_node_decoder(self._type)
    else:
      self._decoder = self._graph.get_edge_decoder(self._type)

  def _add_param(self, name, value):
    self._params[name] = value

  def _add_in_edge(self, edge):
    self._in_edges.append(edge)

  def _add_out_edge(self, edge):
    self._out_edges.append(edge)

  def set_output_field(self, field):
    self._output_field = field

  @property
  def output_field(self):
    return self._output_field

  @property
  def type(self):
    return self._type

  @property
  def shape(self):
    return self._shape

  @property
  def sparse(self):
    return self._sparse

  @property
  def node_def(self):
    return self._node_def

  def get_alias(self):
    return self._alias

  def get_lookup_node(self):
    return self._lookup_node

  def get_degree_nodes(self):
    return self._degree_nodes

  """ GSL APIs """
  def alias(self, alias):
    self._set_alias(alias, temp=False)
    return self

  def batch(self, batch_size):
    assert isinstance(batch_size, int) and batch_size > 0
    self._shape = (batch_size,)
    self._add_param(pywrap.kBatchSize, batch_size)
    self._add_param(pywrap.kEpoch, sys.maxsize >> 32)
    self._add_param(pywrap.kStrategy, "by_order")
    return self

  def shuffle(self, traverse=False):
    strategy_map = {True: "shuffle", False: "random"}
    self._add_param(pywrap.kStrategy, strategy_map.get(traverse, False))
    return self

  def sample(self, count):
    """Sample count of neighbors for each upstream node.

    Args:
      count (int): For "full" strategy, the neighbors will be truncated by
        the given count if `count` > 0, otherwise, the real count of neighbors
        would return. For other sampling strategies, the returned neighbors
        would be padded with the given PaddingMode.
    """
    assert isinstance(count, int)
    self._add_param(pywrap.kNeighborCount, count)
    self._add_param(pywrap.kPartitionKey, pywrap.kSrcIds)
    self._shape = (np.prod(self._shape), count)
    return self

  def by(self, strategy):
    self._strategy = strategy
    if self._op_name == "NegativeSampler":
      assert strategy in ["random", "in_degree", "conditional", "node_weight"]
    elif self._op_name == "Sampler":
      assert strategy in \
        ["random", "topk", "in_degree", "edge_weight", "full"]
    else:
      raise ValueError("`by(strategy)` can only be used after`sample(count)`")

    self._sparse = (strategy == "full")
    self._op_name = strategy2op(self._strategy, self._op_name)
    self._add_param(pywrap.kStrategy, self._op_name)
    return self

  def filter(self, target):
    """Filter the samples that are not equal to target ids.

    Args:
      target (string): Alias of upstream TraverseVertexDagNode.

    Raises:
      ValueError: target upstream is not existed.
    """
    if isinstance(target, str):
      target = self._dag.get_node(target)
    if not isinstance(target, TraverseVertexDagNode):
      raise ValueError("filter only accepts upstream Nodes.")
    edge = self._new_edge(src_output=target.output_field,
                          dst_input=pywrap.kFilterIds)
    target._add_out_edge(edge)
    self._add_in_edge(edge)
    self._add_param(pywrap.kFilterType, 1)
    return self

  def where(self, target, condition={}):
    """ Add condition for negative samlpler. Used after `by`.

    Args:
      target (string): Alias of upstream TraverseVertexDagNode which is the
        postive sample that condition should match.
      condition (dict, optional): Keys are as following.
        "batch_share" (bool, optional): Whether sampled negative samples are
          shared by this batch. Defaults to False.
        "unique" (bool, optional): Whether sampled negtive samples are unique.
          Defaults to False.
        "int_cols" (int list, optional): int columns as condition.
          Defaults to [].
        "int_props" (float list, optional) : proportions of int columns.
          Defaults to [].
        "float_cols" (int list, optional): float columns as condition.
          Defaults to [].
        "float_props" (float list, optional): proportions of float columns.
          Defaults to [].
        "str_cols" (int list, optional): string columns as condition.
          Defaults to [].
        "str_props" (float list, optional): proportions of string columns.
          Defaults to [].

    Raises:
      ValueError: target upstream is not existed.
    """
    if isinstance(target, str):
      target = self._dag.get_node(target)
    if not isinstance(target, TraverseVertexDagNode):
      raise ValueError("where only accepts upstream Nodes.")
    edge = self._new_edge(src_output=target.output_field,
                          dst_input=pywrap.kDstIds)
    target._add_out_edge(edge)
    self._add_in_edge(edge)

    default_kvs = {
      "batch_share": (pywrap.kBatchShare, False),
      "unique": (pywrap.kUnique, False),
      "int_cols": (pywrap.kIntCols, None),
      "int_props": (pywrap.kIntProps, None),
      "float_cols": (pywrap.kFloatCols, None),
      "float_props": (pywrap.kFloatProps, None),
      "str_cols": (pywrap.kStrCols, None),
      "str_props": (pywrap.kStrProps, None)
    }

    for k in condition.keys():
      if k not in default_kvs.keys():
        raise ValueError("condition {} is not supported.".format(k))
    for k, v in default_kvs.items():
      param_key, default_value = v
      value = condition.get(k, default_value)
      if value is not None:
        self._add_param(param_key, value)

    self._op_name = "ConditionalNegativeSampler"
    self._add_param(pywrap.kStrategy, self._strategy)
    self._add_param(pywrap.kDstType, target.type)
    return self

  def each(self, func):
    func(self)
    return self

  def values(self, func=lambda x: x):
    self._dag.set_ready(func)
    return self._dag

  """ GSL Apis """

  def _set_alias(self, alias=None, temp=False):
    if self._alias:
      return
    if not alias:
      alias = str(uuid.uuid1())
    self._alias = alias
    self._dag.add_node(alias, self, temp=temp)

    self._lookup_node = self._lookup()
    self._link_to_sink()

  def _get_shape_and_degrees(self, dag_values):
    shape = self._shape
    degrees = None
    if self._sparse:
      assert isinstance(shape, tuple) and len(shape) == 2
      degrees = pywrap.get_dag_value(dag_values, self._nid, pywrap.kDegreeKey)
      shape = (degrees.size, shape[1] if shape[1] and shape[1] > 0 \
          else max(degrees))
    return shape, degrees

  def feed_values(self, dag_values):
    pass

  def _reverse_edge(self, edge_type, force=True):
    reverse_mask = "_reverse"
    if edge_type.endswith(reverse_mask):
      return edge_type[: -len(reverse_mask)]
    elif force:
      return edge_type + reverse_mask
    return edge_type

  def _new_edge(self, src_output=None, dst_input=None):
    # add an edge for cur node
    eid = get_eid()
    cur_edge = get_dag_edge(eid)
    default_field = "fake"
    cur_edge.src_output = src_output or self._output_field or default_field
    cur_edge.dst_input = dst_input or default_field
    return cur_edge

  def _new_edge_node(self, op_name, edge_type, in_edge):
    assert edge_type is not None and isinstance(edge_type, str)

    self._add_out_edge(in_edge)
    shape = self._shape

    next_node = TraverseEdgeDagNode(self._dag, op_name=op_name,
                                    upstream=self)
    next_node._shape = shape
    next_node._add_param(pywrap.kEdgeType, edge_type)
    next_node._add_in_edge(in_edge)

    next_node.set_path(edge_type, pywrap.NodeFrom.NODE)
    next_node.set_output_field(pywrap.kEdgeIds)
    return next_node

  def _new_vertex_node(self, op_name, edge_type, in_edge,
                       node_from=pywrap.NodeFrom.EDGE_DST):
    assert edge_type is not None and isinstance(edge_type, str)

    self._add_out_edge(in_edge)
    shape = self._shape

    next_node = TraverseVertexDagNode(self._dag, op_name=op_name)
    next_node._shape = shape
    next_node._add_param(pywrap.kEdgeType, edge_type)
    next_node._add_in_edge(in_edge)

    next_node.set_path(edge_type, node_from)
    next_node.set_output_field(pywrap.kNodeIds)
    return next_node

  def set_ready(self, node_id):
    """ Set dag_node ready and format the DagNodeDef proto.
    """
    add_param_map = {
      int: pywrap.add_dag_node_int_params,
      str: pywrap.add_dag_node_string_params,
      unicode: pywrap.add_dag_node_string_params}
    add_vector_param_map = {
      int: pywrap.add_dag_node_int_vector_params,
      float: pywrap.add_dag_node_float_vector_params}

    self._nid = node_id
    node_def = pywrap.new_dag_node()
    pywrap.set_dag_node_id(node_def, node_id)
    pywrap.set_dag_node_op_name(node_def, self._op_name)
    for in_edge in self._in_edges:
      in_edge.dst = self
      pywrap.add_dag_node_in_edge(
        node_def, in_edge.dag_edge_def)
    for out_edge in self._out_edges:
      out_edge.src = self
      pywrap.add_dag_node_out_edge(
        node_def, out_edge.dag_edge_def)

    for k, v in self._params.items():
      if isinstance(v, bool):
        v = int(v)
      if not isinstance(v, list):
        add_param_map[type(v)](node_def, k, v)
        continue
      if len(v) == 0:
        continue
      add_vector_param_map[type(v[0])](node_def, k, v)
    self._node_def = node_def
    return True

  def _lookup(self):
    return None

  def _link_to_sink(self):
    edge = self._new_edge()
    self._add_out_edge(edge)
    self._dag.sink_node._add_in_edge(edge)


class SinkNode(DagNode):
  def __init__(self, dag):
    super(SinkNode, self).__init__(dag, "Sink")
    self._dag.sink_node = self

  def alias(self, alias):
    self._set_alias(alias, temp=True)
    return self

  def _lookup(self):
    # Override
    return None

  def _link_to_sink(self):
    # Override
    pass


class TraverseVertexDagNode(DagNode):
  def __init__(self, dag, op_name="", params={}):
    super(TraverseVertexDagNode, self).__init__(dag, op_name, params)

  def outV(self, edge_type=None):
    self._set_alias()
    in_edge = self._new_edge(dst_input=pywrap.kSrcIds)
    next_node = self._new_vertex_node("Sampler", edge_type, in_edge)
    self._pos_downstreams.append(next_node)
    self._add_degree_node(edge_type, pywrap.NodeFrom.EDGE_SRC)
    return next_node

  def inV(self, edge_type=None):
    self._set_alias()
    in_edge = self._new_edge(dst_input=pywrap.kSrcIds)
    next_node = self._new_vertex_node("Sampler",
                                      self._reverse_edge(edge_type), in_edge)
    self._pos_downstreams.append(next_node)
    self._add_degree_node(self._reverse_edge(edge_type), pywrap.NodeFrom.EDGE_SRC)
    return next_node

  def outE(self, edge_type):
    self._set_alias()
    in_edge = self._new_edge(dst_input=pywrap.kSrcIds)
    next_node = self._new_edge_node("Sampler", edge_type, in_edge)
    self._pos_downstreams.append(next_node)
    return next_node

  def inE(self, edge_type):
    self._set_alias()
    in_edge = self._new_edge(dst_input=pywrap.kSrcIds)
    next_node = self._new_edge_node("Sampler",
                                    self._reverse_edge(edge_type), in_edge)
    self._pos_downstreams.append(next_node)
    return next_node

  def outNeg(self, edge_type):
    self._set_alias()
    in_edge = self._new_edge(dst_input=pywrap.kSrcIds)
    next_node = self._new_vertex_node("NegativeSampler", edge_type, in_edge)
    self._neg_downstreams.append(next_node)
    return next_node

  def inNeg(self, edge_type):
    self._set_alias()
    in_edge = self._new_edge(dst_input=pywrap.kSrcIds)
    next_node = self._new_vertex_node("NegativeSampler",
                                      self._reverse_edge(edge_type), in_edge)
    self._neg_downstreams.append(next_node)
    return next_node

  def Neg(self, node_type):
    self._set_alias()
    in_edge = self._new_edge(dst_input=pywrap.kSrcIds)
    next_node = self._new_vertex_node("NegativeSampler", node_type, in_edge,
                                      node_from=pywrap.NodeFrom.NODE)
    self._neg_downstreams.append(next_node)
    return next_node

  def _lookup(self):
    # Override
    # Generate an edge from traverse node to it's lookup node.
    edge = self._new_edge(dst_input=pywrap.kNodeIds)
    self._add_out_edge(edge)
    return LookupDagNode("LookupNodes", self, [edge],
                         {pywrap.kNodeType: self._type,
                          pywrap.kPartitionKey: pywrap.kNodeIds})

  def feed_values(self, dag_values):
    shape, degrees = self._get_shape_and_degrees(dag_values)
    return self._graph.get_nodes(
      self._type, pywrap.get_dag_value(
        dag_values, self._nid, pywrap.kNodeIds), degrees, shape)


class TraverseNegVertexDagNode(TraverseVertexDagNode):
  def __init__(self, dag, op_name="", params={}):
    super(TraverseNegVertexDagNode, self).__init__(dag, op_name, params)

class TraverseEdgeDagNode(DagNode):
  def __init__(self, dag, op_name="", params={}, upstream=None):
    super(TraverseEdgeDagNode, self).__init__(dag, op_name, params)
    self._upstream = upstream

  def inV(self):
    self._set_alias()
    next_node = FakeNode(self)
    next_node.set_path(self._path, pywrap.NodeFrom.EDGE_DST)
    next_node.set_output_field(pywrap.kNodeIds)
    next_node._sparse = self.sparse
    self._pos_downstreams.append(next_node)
    return next_node

  def outV(self):
    raise warnings.warn("outV is just the upstream Nodes.")

  def _lookup(self):
    # Override
    # Generate an edge from current node to it's lookup node.
    nbr_count = self._params.get(pywrap.kNeighborCount, 0)
    edge = self._new_edge(dst_input=pywrap.kEdgeIds)
    # Add an edge from upstream node to lookup node.
    extra_edge = self._new_edge(
      src_output=self._upstream.output_field, dst_input=pywrap.kSrcIds)
    edges = [edge, extra_edge]
    if self._sparse:
      degree_edge = self._new_edge(src_output=pywrap.kDegreeKey,
                                   dst_input=pywrap.kDegreeKey)
      edges.append(degree_edge)
      self._add_out_edge(degree_edge)
    self._add_out_edge(edge)
    self._upstream._add_out_edge(extra_edge)

    return LookupDagNode(
        "LookupEdges", self, edges,
        {pywrap.kEdgeType: self._type,
         pywrap.kPartitionKey: pywrap.kSrcIds,
         pywrap.kNeighborCount: nbr_count})

  def feed_values(self, dag_values):
    shape, degrees = self._get_shape_and_degrees(dag_values)
    edge_ids = pywrap.get_dag_value(dag_values, self._nid, pywrap.kEdgeIds)
    assert isinstance(shape, tuple) and len(shape) == 2
    src_ids = pywrap.get_dag_value(dag_values,
                                   self._upstream.nid,
                                   self._upstream.output_field)
    dst_ids = pywrap.get_dag_value(dag_values, self._nid, pywrap.kNodeIds)
    nbr_counts = degrees if self._sparse else [shape[1]] * shape[0]
    nbr_counts = nbr_counts[:len(src_ids)]
    src_ids = np.concatenate(
            [src_ids[idx].repeat(d) for idx, d in enumerate(nbr_counts)])
    return self._graph.get_edges(
        self._type, src_ids, dst_ids, edge_ids, degrees, shape)


class TraverseSourceEdgeDagNode(TraverseEdgeDagNode):
  def __init__(self, dag, op_name="", params={}):
    super(TraverseSourceEdgeDagNode, self).__init__(
      dag, op_name=op_name, params=params)

  def outV(self):
    self._set_alias()
    next_node = FakeNode(self)
    next_node.set_path(self._path, pywrap.NodeFrom.EDGE_SRC)
    next_node.set_output_field(pywrap.kSrcIds)
    return next_node

  def inV(self, edge_type=None):
    self._set_alias()
    next_node = FakeNode(self)
    next_node.set_path(self._path, pywrap.NodeFrom.EDGE_DST)
    next_node.set_output_field(pywrap.kDstIds)
    return next_node

  def _lookup(self):
    # Override
    edge = self._new_edge(dst_input=pywrap.kEdgeIds)
    extra_edge = self._new_edge(src_output=pywrap.kSrcIds,
                                dst_input=pywrap.kSrcIds)
    self._add_out_edge(edge)
    self._add_out_edge(extra_edge)

    return LookupDagNode(
        "LookupEdges", self, [edge, extra_edge],
        {pywrap.kEdgeType: self._type,
         pywrap.kPartitionKey: pywrap.kSrcIds})

  def feed_values(self, dag_values):
    shape, degrees = self._get_shape_and_degrees(dag_values)
    edge_ids = pywrap.get_dag_value(dag_values, self._nid, pywrap.kEdgeIds)
    src_ids = pywrap.get_dag_value(dag_values, self._nid, pywrap.kSrcIds)
    dst_ids = pywrap.get_dag_value(dag_values, self._nid, pywrap.kDstIds)
    return self._graph.get_edges(
        self._type, src_ids, dst_ids, edge_ids, degrees, shape)


class LookupDagNode(DagNode):
  def __init__(self, op_name="", upstream=None, in_edges=[], params={}):
    super(LookupDagNode, self).__init__(upstream._dag, op_name, params)
    self._upstream = upstream
    self._shape = upstream._shape
    for edge in in_edges:
      self._add_in_edge(edge)
    self._set_alias(temp=True)
    self.set_output_field("properties")


class DegreeDagNode(DagNode):
  def __init__(self, op_name="", upstream=None, in_edges=[], params={}):
    super(DegreeDagNode, self).__init__(upstream._dag, op_name, params)
    self._upstream = upstream
    self._shape = upstream._shape
    for edge in in_edges:
      self._add_in_edge(edge)
    self._set_alias(temp=True)
    self.set_output_field(pywrap.kDegrees)

  @property
  def edge_type(self):
    return self._params[pywrap.kEdgeType]

  @property
  def node_from(self):
    return self._params[pywrap.kNodeFrom]


class FakeNode(TraverseVertexDagNode):
  """ FakeNode is used for adding corresponding DagNode of E.outV()/inV()
  to Dag. E.outV()/inV() doesn't raise any operator, but only changes the
  field which downstream absorbs.
  """
  def __init__(self, dag_node):
    super(FakeNode, self).__init__(dag_node._dag)
    if not isinstance(dag_node, TraverseEdgeDagNode):
      raise ValueError("FakeNode is a fake for TraverseEdgeDagNode, not {}"
                       .format(type(dag_node)))
    self._upstream = dag_node
    self._shape = dag_node._shape

  @property
  def nid(self):
    return self._upstream.nid

  def set_ready(self, node_id):
    # Override
    return False

  def _add_out_edge(self, edge):
    # Override
    self._upstream._add_out_edge(edge)

  def _link_to_sink(self):
    # Override
    pass

  def feed_values(self, dag_values):
    edges = self._upstream.feed_values(dag_values)
    return (edges.dst_nodes, edges.src_nodes)[int(self._output_field == pywrap.kSrcIds)]
