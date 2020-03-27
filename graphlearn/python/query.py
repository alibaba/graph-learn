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
""" Query Engine for Gremlin-like API.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types
import warnings
from enum import Enum

import numpy as np

from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python.values import Nodes, Edges


class _SamplerType(Enum):  # pylint: disable=invalid-name
  """ The action on the current path, inculdes sample and get value from
  ids or generator. If sample, the _SamplerType indicates the sample type,
  which includes sample node, sample edge, sample neighbor and sampler negative
  neighbor.
  """
  gen = 1
  ids = 2
  node = 3
  edge = 4
  neighbor = 5
  negative = 6


class _TraversalPath(Enum):  # pylint: disable=invalid-name
  """ The travelsal path on the graph.
  """
  V = 1
  E = 2
  in_ = 3
  out = 4
  inE = 5  # pylint: disable=invalid-name
  outE = 6  # pylint: disable=invalid-name
  inNeg = 7  # pylint: disable=invalid-name
  outNeg = 8  # pylint: disable=invalid-name
  inV = 9  # pylint: disable=invalid-name
  outV = 10  # pylint: disable=invalid-name


class Query(object):
  """ Base Query Node to construct query tree. All the steps are
  to generate Query.
  """

  def __init__(self, query_engine):
    self.qe = query_engine
    self.sampler_count = None
    self.sampler_path = None
    self.sampler_type = None
    self.sampler_strategy = "by_order"
    self.sampler_args = {}
    self.cur_node = None
    self.query_root = None
    self.traversal_path = None
    self.traversal_node = None
    self.name = None

  def batch(self, batch_size):
    self.sampler_count = batch_size
    if not self.cur_node:
      self.cur_node = _QueryNode(self)
      self.query_root = self.cur_node
    else:
      warnings.warn(".batch not work.")
    return self

  def sample(self, count=None):
    """ How many neighbor edges or nodes will be sampled
    for each seed node.
    """
    if not count:
      count = 1
    self.sampler_count = count
    return self

  def by(self, strategy, **kwargs):
    """ Define the strategy of the sample.
    """
    self.sampler_strategy = strategy
    self.sampler_args.update(kwargs)
    next_node = _QueryNode(self)
    self.cur_node.add_child(next_node)
    self.cur_node = next_node
    return self

  def alias(self, name):
    """ We call each *V()/*E()/*Neg() a traverlsal step.
    If no traversal step has alias, all the results of traversal step
    are returned in a list, sort by traversal order.
    If one or more traversal step have alias, the results will be
    returned in a map, whose keys are the aliases. Then the traversal step
    should be get by the alias key.
    """
    self.cur_node.name = name
    return self

  def repeat(self, func, times, alias_list=None, params_list=None):
    """ repeat query fragment N times.

    Args:
      func: a python function. input and output should be the
        same type of Query.
      times: integer. How many times the function whould repeat.
      alias_list: list. Default is None, the whole query should
        without alias so that the result returned as a list.
        Or alias each repeated step.

    Example:
    >>> func = lambda v: v.outV("i-i").sample(5).by("random")
    >>> q = g.V("u-i").batch(64).repeat(func, 3).values()
    >>> q.run(q) # return a list of Nodes,
    >>>          # shapes are [[64], [64, 5], [64*5, 5], [64*5*5, 5]]
    >>> # a wrong func:
    >>> func = lambda v: v.outE("i-i").sample(5).by("random")
    >>> q = g.V("u-i").batch(64).repeat(func, 3).values()
    >>> # func input with a VertexQuery and output with an EdgeQuery.
    """
    if not params_list:
      if not alias_list:
        for i in range(times):
          self = func(self)
      else:
        if len(alias_list) != times:
          raise ValueError("length of alias_list must be the same with times.")
        for i in range(times):
          self = func(self).alias(alias_list[i])
    else:
      if not alias_list:
        for i in range(times):
          self = func(self, params=params_list[i])
      else:
        if len(alias_list) != times:
          raise ValueError("length of alias_list must be the same with times.")
        for i in range(times):
          self = func(self, params=params_list[i]).alias(alias_list[i])
    return self

  def each(self, func):
    """ Feed the upstream Query to multiple downstream Queries.
    If each is used, we should call alias for each step we want to return
    so that we can distinguish every step.
    Args:
      func: a python function. Input a Query and output a list of Queries.
    """
    func(self)
    return self

  def values(self, func=None):
    """Describe how to process the sampled data in the dict, or
    in the list when no step with alias. If func is None, just return
    the generator of original dict or list.
    Args:
      func: python function. Input is a map or a list(no alias exists),
        output can be user-defined. Default is None, which means return
        the original Nodes and Edges.
    >>> # Example 1: emit a list of Nodes
    >>> q = g.V("u-i").batch(3) \
    >>>  .out("u-i").sample(5).by("edge_weight") \
    >>>  .values() # return a data generator
    >>> g.run(q) # [Nodes, Nodes], shape = [[3], [3, 5]]
    """
    gen = self.qe.parse_tree(self.query_root, func)
    return gen

  def emit(self, func=None, **kwargs):
    """The combination of Query.values() and Graph.run(),
    return the exact dict or list.
    Args:
      func: python function. Input is a map or a list(no alias exists),
        output can be user-defined. Default is None, which means return
        the original Nodes and Edges.
    Example:
    >>> # Example 1: emit a list of Nodes
    >>> g.V("user", ids=np.array([1, 2, 3])) \
    >>>  .out("u-i").sample(5).by("edge_weight") \
    >>>  .emit() # [Nodes, Nodes], shape = [[3], [3, 5]]
    """
    gen = self.qe.parse_tree(self.query_root, func)
    return gen.next(**kwargs)

  def _sub_vertex_query(self, qe, node_type):
    sub_query = VertexQuery(qe, node_type)
    sub_query.cur_node = self.cur_node
    sub_query.query_root = self.query_root
    return sub_query

  def _sub_edge_query(self, qe, edge_type):
    sub_query = EdgeQuery(qe, edge_type)
    sub_query.cur_node = self.cur_node
    sub_query.query_root = self.query_root
    return sub_query

  def _get_node_types_in_edge(self, edge_type):
    """ Get src_type and dst_type of the edge_type.
    """
    topology = self.qe.graph.get_topology()
    src_type, dst_type = topology.get_src_type(edge_type), \
                         topology.get_dst_type(edge_type)
    return src_type, dst_type

  def _make_path(self, sub_query):
    """ Construct sub_query with sampler path and traversal vertex.
    """
    edge_type = sub_query.sampler_path
    src_type, dst_type = self._get_node_types_in_edge(edge_type)
    if self.traversal_vertex == src_type and \
        sub_query.traversal_path in \
        (_TraversalPath.out, _TraversalPath.outE, _TraversalPath.outNeg):
      sub_query.sampler_path = edge_type
      sub_query.traversal_vertex = dst_type
    elif self.traversal_vertex == dst_type and \
        sub_query.traversal_path in \
        (_TraversalPath.in_, _TraversalPath.inE, _TraversalPath.inNeg):
      if self.qe.graph.is_directed(edge_type):
        raise ValueError("edge_type {} is not undirected, you should not "
                         "use `inE` or `inNeg`.".format(edge_type))
      if src_type != dst_type:
        sub_query.sampler_path = edge_type + "_reverse"
      else:
        sub_query.sampler_path = edge_type
      sub_query.traversal_vertex = src_type
    else:
      if self.traversal_vertex not in (src_type, dst_type):
        raise ValueError("{} is not the source or destination node of "
                         "edge type of {}.".format(self.traversal_vertex,
                                                   edge_type))
      else:
        raise ValueError("`out*` can only be used after src_type of the edge,"
                         "in*` can only be used after dst_type of the egde.")
    return sub_query


class VertexQuery(Query):
  """ A Query class that has all vertex traversal functions.
  """

  def __init__(self, query_engine, id_type, feed=None, **kwargs):
    """ Query Node on vertex to construct query tree.
    """
    super(VertexQuery, self).__init__(query_engine)

    self.traversal_path = _TraversalPath.V
    self.sampler_path = id_type
    node_from = kwargs.get("node_from", pywrap.NodeFrom.NODE)
    if node_from == pywrap.NodeFrom.NODE:
      self.traversal_vertex = id_type
    else:
      if not self._is_edge_type_existed(id_type, query_engine):
        raise ValueError("edge_type {} not exist.".format(id_type))
      src_type, dst_type = self._get_node_types_in_edge(id_type)
      if node_from == pywrap.NodeFrom.EDGE_DST:
        self.traversal_vertex = dst_type
      elif node_from == pywrap.NodeFrom.EDGE_SRC:
        self.traversal_vertex = src_type
      else:
        raise ValueError("node_from must be gl.EDGE_SRC or gl.EDGE_DST"
                         " if feed g.V() with edge type.")
    self.data = feed
    if self.data is not None:
      self.sampler_type = self.data
      self.cur_node = _QueryNode(self)
      self.query_root = self.cur_node
    else:
      self.sampler_type = _SamplerType.node
    self.sampler_args.update(kwargs)

  def _is_edge_type_existed(self, edge_type, query_engine):
    topology = query_engine.graph.get_topology()
    return topology.is_exist(edge_type)

  def shuffle(self, traverse=False):
    if traverse:
      self.sampler_strategy = "shuffle"
    else:
      self.sampler_strategy = "random"
    return self

  def outV(self, edge_type):  # pylint: disable=invalid-name
    """ Traverse to the destination node along the edge.
    For exmaple, the topology is nodeA--(edge)-->nodeB,
    nodeA.outV(edge) is nodeB.
    """
    sub_query = self._sub_vertex_query(self.qe, edge_type)
    sub_query.sampler_type = _SamplerType.neighbor
    sub_query.traversal_path = _TraversalPath.out

    sub_query = self._make_path(sub_query)
    return sub_query

  def inV(self, edge_type):  # pylint: disable=invalid-name
    """ Traverse to the source node along the edge.
    For exmaple, the topology is nodeA--(edge)-->nodeB,
    nodeB.inV(edge) is nodeA.
    """
    sub_query = self._sub_vertex_query(self.qe, edge_type)
    sub_query.sampler_type = _SamplerType.neighbor
    sub_query.traversal_path = _TraversalPath.in_
    sub_query = self._make_path(sub_query)
    return sub_query

  def outE(self, edge_type):  # pylint: disable=invalid-name
    """ Traverse to the edge whose source node is the current node.
    For exmaple, the topology is nodeA--(edge)-->nodeB,
    nodeA.outE(edge) is edge.
    """
    sub_query = self._sub_edge_query(self.qe, edge_type)
    sub_query.sampler_type = _SamplerType.neighbor
    sub_query.traversal_path = _TraversalPath.outE
    sub_query = self._make_path(sub_query)
    return sub_query

  def inE(self, edge_type):  # pylint: disable=invalid-name
    """ Traverse to the edge whose dst node is the current node.
    For exmaple, the topology is nodeA--(edge)-->nodeB,
    nodeB.inE(edge) is edge.
    """
    sub_query = self._sub_edge_query(self.qe, edge_type)
    sub_query.sampler_type = _SamplerType.neighbor
    sub_query.traversal_path = _TraversalPath.inE
    sub_query = self._make_path(sub_query)
    return sub_query

  def outNeg(self, edge_type):  # pylint: disable=invalid-name
    """ Traverse to the dst negative node along the edge.
    For exmaple, the topology is nodeA--(edge)-->nodeB,
    nodeA.outNeg(edge) is nodeB.
    """
    sub_query = self._sub_vertex_query(self.qe, edge_type)
    sub_query.sampler_type = _SamplerType.negative
    sub_query.traversal_path = _TraversalPath.outNeg
    sub_query = self._make_path(sub_query)
    return sub_query

  def inNeg(self, edge_type):  # pylint: disable=invalid-name
    """ Traverse to the source negative node along the edge.
    For exmaple, the topology is nodeA--(edge)-->nodeB,
    nodeB.outNeg(edge) is nodeA.
    """
    sub_query = self._sub_vertex_query(self.qe, edge_type)
    sub_query.sampler_type = _SamplerType.negative
    sub_query.traversal_path = _TraversalPath.inNeg
    sub_query = self._make_path(sub_query)
    return sub_query

  def Neg(self, node_type):  # pylint: disable=invalid-name
    sub_query = self._sub_vertex_query(self.qe, node_type)
    sub_query.sampler_type = _SamplerType.negative
    sub_query.traversal_path = _TraversalPath.outNeg
    sub_query.sampler_path = node_type
    sub_query.traversal_vertex = node_type
    return sub_query


class EdgeQuery(Query):
  """ A Query class that has all edge traversal functions.
  """

  def __init__(self, query_engine, edge_type, feed=None):
    """ Query Node on edge to construct query tree.
    """
    super(EdgeQuery, self).__init__(query_engine)

    self.traversal_path = _TraversalPath.E
    self.sampler_path = edge_type
    self.data = feed
    if self.data is not None:
      self.sampler_type = self.data
      self.cur_node = _QueryNode(self)
      self.query_root = self.cur_node
    else:
      self.sampler_type = _SamplerType.edge

  def shuffle(self, traverse=False):
    if traverse:
      self.sampler_strategy = "shuffle"
    else:
      self.sampler_strategy = "random"
    return self

  def outV(self):  # pylint: disable=invalid-name
    """ Traverse to the node which the current edge's arrow from.
    For exmaple, the topology is nodeA--(edge)-->nodeB,
    edge.outV(edge) is nodeA.
    """
    sub_query = self._sub_vertex_query(self.qe, None)
    sub_query.traversal_path = _TraversalPath.outV
    src_type, _ = self._get_node_types_in_edge(self.sampler_path)
    sub_query.traversal_vertex = src_type
    next_node = _QueryNode(sub_query)
    sub_query.cur_node.add_child(next_node)
    sub_query.cur_node = next_node
    return sub_query

  def inV(self):  # pylint: disable=invalid-name
    """ Traverse to the node which the current edge's arrow points to.
    For exmaple, the topology is nodeA--(edge)-->nodeB,
    edge.inV(edge) is nodeB.
    """
    sub_query = self._sub_vertex_query(self.qe, None)
    sub_query.traversal_path = _TraversalPath.inV
    _, dst_type = self._get_node_types_in_edge(self.sampler_path)
    sub_query.traversal_vertex = dst_type
    next_node = _QueryNode(sub_query)
    sub_query.cur_node.add_child(next_node)
    sub_query.cur_node = next_node
    return sub_query


class QueryEngine(object):
  """ QueryEngine is to parse query tree.
  """

  def __init__(self, graph):
    self._graph = graph

  @property
  def graph(self):
    return self._graph

  def parse_tree(self, query_root, func):
    """parse query tree to generator and samplers, and record the link
    between them.
    """
    cur_query = query_root.value
    source = None
    if isinstance(cur_query.sampler_type, (np.ndarray, tuple, Nodes, Edges)):
      data_func = lambda x: x  # pylint: disable=unnecessary-lambda
    elif isinstance(cur_query.sampler_type, types.GeneratorType):
      data_func = lambda x: next(x)  # pylint: disable=unnecessary-lambda
    else:
      source = getattr(self._graph, cur_query.sampler_type.name + '_sampler') \
        (cur_query.sampler_path,
         cur_query.sampler_count,
         strategy=cur_query.sampler_strategy,
         **cur_query.sampler_args)
    if not source:
      if cur_query.traversal_path == _TraversalPath.V:
        source = _NodeGen(self._graph, cur_query.sampler_type,
                          cur_query.sampler_path, data_func)
      elif cur_query.traversal_path == _TraversalPath.E:
        source = _EdgeGen(self._graph, cur_query.sampler_type,
                          cur_query.sampler_path, data_func)
      else:
        raise ValueError("Wrong traversal_path {}"
                         .format(cur_query.traversal_path))
    queue = [query_root]
    parsed_root = _ParsedNode(_PositionedSampler(_NamedSource(source,
                                                              query_root.name),
                                                 cur_query.traversal_path))
    parsed_queue = [parsed_root]
    while queue:
      node = queue.pop(0)
      cur_parsed = parsed_queue.pop(0)
      for child in node.children:
        queue.append(child)
        child_value = child.value
        if child_value.sampler_path:
          s = getattr(self._graph, child_value.sampler_type.name + '_sampler') \
            (child_value.sampler_path, child_value.sampler_count,
             strategy=child_value.sampler_strategy,
             **child_value.sampler_args)
        else:
          s = None
        parsed = _ParsedNode(_PositionedSampler(_NamedSource(s, child.name),
                                                child_value.traversal_path))
        parsed_queue.append(parsed)
        cur_parsed.add_child(parsed)
    gen = _ValueIter(parsed_root, self._graph, func)
    return gen


class _TreeNode(object):  # pylint: disable=invalid-name
  def __init__(self, value):
    self.value = value
    self.children = []

  def add_child(self, query_node):
    self.children.append(query_node)


class _QueryNode(_TreeNode):  # pylint: disable=invalid-name
  def __init__(self, value):
    super(_QueryNode, self).__init__(value)
    self.name = None


class _ParsedNode(_TreeNode):  # pylint: disable=invalid-name
  def __init__(self, value):
    super(_ParsedNode, self).__init__(value)
    self.target = None


class _ValueIter(object):  # pylint: disable=invalid-name
  """ An iteration that return sampled values.
  """

  def __init__(self, parsed_root, graph, func):
    self.parsed_root = parsed_root
    self.func = func
    self.graph = graph

  def __iter__(self):
    return self

  def next(self, **kwargs):
    """ Get the next batch of sampled values.
    """
    res = {}
    res_list = []
    cur = self.parsed_root
    queue = [cur]
    while queue:
      node = queue.pop(0)
      value, position = node.value.named_src, node.value.position
      value_src, value_name = value.source, value.name
      last = node.target
      if position == _TraversalPath.E or position == _TraversalPath.V:
        last = value_src.get()
      elif position == _TraversalPath.outV:
        last = last.src_nodes
      elif position == _TraversalPath.inV:
        last = last.dst_nodes
      elif position in (_TraversalPath.outE, _TraversalPath.inE):
        last = value_src.get(last.ids, **kwargs).layer_edges(1)
      elif position in (_TraversalPath.out, _TraversalPath.in_):
        last = value_src.get(last.ids, **kwargs).layer_nodes(1)
      elif position in (_TraversalPath.outNeg, _TraversalPath.inNeg):
        last = value_src.get(last.ids, **kwargs)
      else:
        raise NotImplementedError("{} not supported for now."
                                  .format(position))
      if value_name:
        res[value_name] = last
      res_list.append(last)
      for child in node.children:
        child.target = last
      queue.extend(node.children)
    if not res:
      if len(res_list) == 1:
        res_list = res_list[0]
      if not self.func:
        return res_list
      return self.func(res_list)
    if not self.func:
      return res
    return self.func(res)


class _NamedSource(object):  # pylint: disable=invalid-name
  def __init__(self, source, name):
    self.source = source
    self.name = name


class _PositionedSampler(object):  # pylint: disable=invalid-name
  def __init__(self, named_src, position):
    self.named_src = named_src
    self.position = position


class _ValueGen(object):  # pylint: disable=invalid-name
  def __init__(self):
    pass

  def get(self):
    pass


class _NodeGen(object):  # pylint: disable=invalid-name
  """ Node generator. Should remain `get` function same name and parameters
  with `get` function of Samplers.
  """

  def __init__(self, graph, data, node_type, func):
    self.graph = graph
    self.data = data
    self.node_type = node_type
    self.func = func

  def get(self):
    if isinstance(self.data, Nodes):
      self.data.graph = self.graph
      return self.data
    return self.graph.get_nodes(self.node_type, self.func(self.data))


class _EdgeGen(object):  # pylint: disable=invalid-name
  """ Node generator. Should remain `get` function same name and parameters
  with `get` function of Samplers.
  """

  def __init__(self, graph, data, edge_type, func):
    self.graph = graph
    self.data = data
    self.edge_type = edge_type
    self.func = func

  def get(self):
    if isinstance(self.data, Edges):
      self.data.graph = self.graph
      return self.data
    src_ids, dst_ids = self.func(self.data)
    return self.graph.get_edges(self.edge_type, src_ids, dst_ids)
