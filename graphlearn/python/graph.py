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
""" Entry of GraphLearn.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import json
import warnings

from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python import sampler as samplers
from graphlearn.python.decoder import Decoder
from graphlearn.python.errors import raise_exception_on_not_ok_status
from graphlearn.python.query import VertexQuery, EdgeQuery, QueryEngine
from graphlearn.python.server import Server
from graphlearn.python.topology import Topology
from graphlearn.python.values import \
    Nodes, Edges, SparseNodes, SparseEdges, Values
from graphlearn.python.utils import strategy2op


class Graph(object):
  """ A Graph object maintains a single graph with same edges or heterogenous
  graph with different edges.

  APIs:
    node() & edge(): to make the graph topology.
    init(): init the graph.
    get_topology(): to get the graph topology.
    get_nodes() & get_edges(): to get specified Nodes or Edges from the graph.
    node_sampler() & edge_sampler() & neighbor_sampler() and negative_sampler():
      to sample `Edges`, `Nodes` or `Layers` object from the graph.
  Gemlin-like APIs:
    node() & edge(): to make the graph topology.
    init(): init the graph.
    V() & E(): get the source nodes or edges, start of the gremlin-like query.
  """

  def __init__(self):
    # list of NodeSource added by .node()
    self._node_sources = []
    # list of EdgeSource added by .edge()
    self._edge_sources = []
    # list of NodeSource added by .node()

    # maintain the graph's static topology for fast query.
    self._topology = Topology()
    # maintain a map of node_type with it's decoder
    self._node_decoders = {}
    # maintain a map of edge_type with it's decoder
    self._edge_decoders = {}
    self._undirected_edges = []
    self._query_engine = QueryEngine(self)

    self._deploy_mode = 0
    self._server = None
    self._client = None

    def stop():
      if self._client and self._deploy_mode == 1:
        self._client.stop()
      if self._server:
        self._server.stop()

    atexit.register(stop)

  def node(self,
           source,
           node_type,
           decoder):
    """ Add graph nodes that will be loaded from a given path or an object.

    Args:
      source (string): Data source of path where graph nodes load from.
      node_type (string): Indicates the type of the added nodes.
    """
    if not isinstance(source, str):
      raise ValueError('source for node() must be string.')
    if not isinstance(decoder, Decoder):
      raise ValueError('decoder must be an instance of `Decoder`, got {}'
                       .format(type(decoder)))
    self._node_decoders[node_type] = decoder
    node_source = self._construct_node_source(source, node_type, decoder)
    self._node_sources.append(node_source)
    return self

  def edge(self,
           source,
           edge_type,
           decoder=None,
           directed=True):
    """ Add graph edges that will be loaded from a given path or an object.

    Args:
      source (string): Data source of path where graph edges load from.
      edge_type (tuple): Indicates the src_type, dst_type and edge_type of
        the added edges. The source should only contains one type of edges,
        and the `edge_type` param should be a 3 element tuple that indicates
        the (src_type, dst_type, edge_type) of the edges.
      decoder (`Decoder` object, Optional): Indicates how to parse the data
        source. Default is None, which means edges should have
        (src_ids, dst_ids).
      directed (boolean, Optional): Indicates that i fth edges are directed.
        Default True means directed.
    """
    if not decoder:
      decoder = Decoder()
    if not isinstance(source, str):
      raise ValueError('source for edge() must be a string.')
    if not isinstance(decoder, Decoder):
      raise ValueError('decoder must be an instance of Decoder, got {}'
                       .format(type(decoder)))
    if not isinstance(edge_type, tuple) or len(edge_type) != 3:
      raise ValueError("edge_type must be a tuple of "
                       "(src_type, dst_tye, edge_type).")

    self._edge_decoders[edge_type[2]] = decoder

    self._topology.add(edge_type[2], edge_type[0], edge_type[1])
    edge_source = \
        self._construct_edge_source(source, edge_type,
                                    decoder,
                                    direction=pywrap.Direction.ORIGIN)
    self._edge_sources.append(edge_source)

    if not directed:
      self._undirected_edges.append(edge_type[2])
      if (edge_type[0] != edge_type[1]):  #pylint: disable=superfluous-parens
        edge_source_reverse = \
            self._construct_edge_source(source,
                                        (edge_type[1], edge_type[0],
                                         edge_type[2] + '_reverse'),
                                        decoder,
                                        direction=pywrap.Direction.REVERSED)
        self._edge_sources.append(edge_source_reverse)
        self._edge_decoders[edge_type[2] + "_reverse"] = decoder
        self._topology.add(edge_type[2] + '_reverse',
                           edge_type[1], edge_type[0])
      else:
        edge_source_reverse = \
            self._construct_edge_source(source,
                                        edge_type,
                                        decoder,
                                        direction=pywrap.Direction.REVERSED)
        self._edge_sources.append(edge_source_reverse)
    return self

  def init(self, cluster="", job_name="", task_index=0, **kwargs):
    """ Initialize the graph with creating graph server instance with
    given cluster env info.

    Args:
      cluster (dict | josn str): Empty dict or string when Graph runs with
        local mode. Otherwise, cluster includes server_count, client_count
        and traker.
        server_count (int): count of servers.
        client_count (int): count of clients.
        traker (str): traker path.
      job_name (str): `client` or `server`, default empty means Graph runs
        with local mode.
      task_index (int): index of current server of client.
    """
    tracker = ""
    if not cluster:
      self._deploy_mode = 0
      pywrap.set_deploy_mode(self._deploy_mode)
      self._client = pywrap.in_memory_client()
      task_index = 0
      server_count = 1
    else:
      self._deploy_mode = 1
      pywrap.set_deploy_mode(self._deploy_mode)
      if isinstance(cluster, dict):
        cluster_spec = cluster
      elif isinstance(cluster, str):
        cluster_spec = json.loads(cluster)
      server_count = cluster_spec.get("server_count")
      client_count = cluster_spec.get("client_count")
      if not server_count or not client_count:
        raise ValueError("cluster is composed of server_count,"
                         "worker_count and tracker")
      tracker = cluster_spec.get("tracker")

      pywrap.set_server_count(server_count)
      pywrap.set_client_count(client_count)
      if tracker:
        pywrap.set_tracker(tracker)

    if job_name == "client":
      pywrap.set_client_id(task_index)
      self._client = pywrap.rpc_client()
      self._server = None
    else:
      if job_name == "server":
        self._client = None
      if not tracker and kwargs.get("tracker"):
        tracker = kwargs["tracker"]
      if tracker:
        self._server = Server(task_index, server_count, tracker)
      else:
        self._server = Server(task_index, server_count)
      self._server.start()
      self._server.init(self._edge_sources, self._node_sources)
    return self

  def V(self, t, feed=None, node_from=pywrap.NodeFrom.NODE):  # pylint: disable=invalid-name
    """ Entry of Gremlin-like query. Start from node.

    Args:
      t (string): The type of node which is the entry of query or the type
        of edge when node is from edge source or dst.
      feed (None| numpy.ndarray | types.GeneratorType | `Nodes`): When `feed`
        is not `None`, the `type` should be a node type, which means query the
        attributes of the specified node ids.
        None: Default. Sample nodes with the following .shuffle and .batch API.
        numpy.ndarray: Any shape of ids. Get nodes of the given ids and
          node_type.
        types.Generator: A generator of numpy.ndarray. Get nodes of generated
          ids and given node_type.
        `Nodes`: A `Nodes` object.
      node_from (NodeFrom): Default is `NodeFrom.NODE`, which means sample or
        or iterate node from node. `NodeFrom.EDGE_SRC` means sample or
        iterate node from source node of edge, and `NodeFrom.EDGE_DST` means
        sample or iterate node from destination node of edge. If node is from
        edge, the `type` must be an edge type.

    Return:
      A 'Query' object.

    Example:
    >>> import numpy as np
    >>> g.V("user").shuffle().batch(64)
    >>> g.V("user", feed=np.array([1, 2, 3]))
    >>> def gen():
    >>>   while True:
    >>>     yield  np.array([1, 2, 3])
    >>> gen = gen()
    >>> g.V("user", feed=gen)
    """
    return VertexQuery(self._query_engine, t,
                       feed=feed, node_from=node_from)

  def E(self, edge_type, feed=None, reverse=False):  # pylint: disable=invalid-name
    """ Entry of Gremlin-like query. Start from edge.

    Args:
      edge_type (string): The type of edge which is the entry of query.
      feed (None| (np.ndarray, np.ndarray) | types.GeneratorType | `Edges`):
        None: Default. Sample edges with the following .shuffle and .batch API.
          (np.ndarray, np.ndarray): src_ids, dst_ids. Get edges of the given
          (src_ids, dst_ids) and given edge_type. src_ids and dst_ids must be
          the same shape, dtype is int.
        types.Generator: A generator of (numpy.ndarray, numpy.ndarray). Get
          edges of generated (src_ids, dst_ids) and given edge_type.
        `Edges`: An `Edges` object.

    Return:
      A 'Query' object.

    Example:
    >>> import numpy as np
    >>> g.E("buy").shuffle().batch(64)
    >>> g.E("buy", feed=(np.array([1, 2, 3]), np.array([4, 5, 6]))
    >>> def gen():
    >>>   while True:
    >>>     yield  (np.array([1, 2, 3]), np.array([4, 5, 6]))
    >>> gen = gen()
    >>> g.E("buy", feed=gen)
    """
    if reverse:
      edge_type = edge_type + "_reverse"
    return EdgeQuery(self._query_engine, edge_type, feed=feed)

  def run(self, q, **kwargs):
    """ Run the query, get the result.

    Args:
      q (Query): A query starts from .V()/.E() and ends up with .values().
      args : Remained args that used by sample.get()

    Example:
    >>> q = g.V("user").batch(64).values(lambda x: x)
    >>> for i in range(10):
    >>>   g.run(q)
    """
    return q.next(**kwargs)

  def get_topology(self):
    """ Get the topology of the graph.

    Return:
      `Topology` object.
    """
    return self._topology

  def get_node_decoder(self, node_type):
    """ Get decoder of the specific node_type.
    """
    decoder = self._node_decoders.get(node_type)
    if not decoder:
      warnings.warn("Node_type {} not exist in graph. Use default decoder."
                    .format(node_type))
      decoder = Decoder()
    return decoder

  def get_edge_decoder(self, edge_type):
    """ Get decoder of the specific edge_type.
    """
    decoder = self._edge_decoders.get(edge_type)
    if not decoder:
      warnings.warn("Edge_type {} not exist in graph. Use default decoder."
                    .format(edge_type))
      decoder = Decoder()
    return decoder

  def get_node_decoders(self):
    """ Get all of the node_decoders.
    """
    return self._node_decoders

  def get_edge_decoders(self):
    """ Get all of the edge_decoders.
    """
    return self._edge_decoders

  def get_nodes(self, node_type, ids, offsets=None, shape=None):
    '''Get nodes from the graph.

    Args:
      node_type (string): Type of nodes, which should has been added by
        `Graph.node()`.
      ids (numpy.ndarrayy): ids of nodes. In sparse case, it must be 1D.
      offsets: (list): To get `SparseNodes`, whose dense shape is 2D,
        `offsets` indicates the number of nodes for each line.
        Default None means it is a dense `Nodes`.
      shape (tuple, Optional): Indicates the shape of nodes ids, attrs, etc.
        For dense case, default None means ids.shape. For sparse case, it
        must has a value which indicates the 2D dense shape.

    Return:
      A `Nodes` object or a `SparseNodes` object.
    '''
    if offsets is None:
      nodes = Nodes(ids, node_type, shape=shape, graph=self)
    else:
      nodes = SparseNodes(ids, offsets, shape, node_type, graph=self)
    return nodes

  def get_edges(self, edge_type, src_ids, dst_ids, offsets=None,
                shape=None, reverse=False):
    """ Get edges from the graph.

    Args:
      edge_type (string): Edge type of edges.
      src_ids (numpy.ndarray): Source ids of edges.
      dst_ids (numpy.ndarray): Destination ids of edges.
      offsets: (list): To get `SparseEdges`, whose dense shape is 2D,
        `offsets` indicates the number of edges for each line.
        Default None means it is a dense `Edges`.
      shape (tuple): Indicates the shape of edge src_ids, dst_ids,
        weights, etc. For dense case, default None means src_ids.shape.
        For sparse case, it must have a value which indicates the
        2D dense shape.
      reverse (boolean): Indicates that if the edges are return as reversed.
        Default `False` means return the original edges. `True` can be assigned
        only if the edges are added to the graph as undirected.

    Return:
      An `Edges` object or a `SparseEdges` object.
    """
    if reverse:
      edge_type = edge_type + "_reverse"

    src_type, dst_type = self._topology.get_src_type(edge_type), \
                         self._topology.get_dst_type(edge_type)
    if offsets is None:
      edges = Edges(src_ids, src_type, dst_ids,
                    dst_type, edge_type, shape=shape, graph=self)
    else:
      edges = SparseEdges(src_ids, src_type, dst_ids,
                          dst_type, edge_type, offsets, shape, graph=self)
    return edges

  def is_directed(self, edge_type):
    """ The specific edge_type of edges is directed or not.
    """
    decoder = self._edge_decoders.get(edge_type)
    if not decoder:
      raise ValueError("edge type {} not exist in graph.".format(edge_type))
    if edge_type in self._undirected_edges:
      return False
    return True

  def node_sampler(self,
                   t,
                   batch_size=64,
                   strategy="by_order",
                   node_from=pywrap.NodeFrom.NODE):
    """ Sampler for sample one type of nodes.

    Args:
      t (string): Sample nodes of the given type `t`. `t` can be node type
        which indicates that sampling node dat, otherwise, `t` can be
        edge type which indicate that sampling source node or dst node of
        edge data.
      batch_size (int, Optional): How many nodes will be returned for get().
      strategy (string, Optional): Indicates how to sample edges,
        "by_order", "random" and "shuffle" are supported.
        "by_order": Get node by order. Raise `graphlearn.OutOfRangeError` when
          all the nodes are visited. Each node will be visited and only be
          visited once.
        "random": Randomly get nodes. No visting state will be hold, so out of
          range will not happened.
        "shuffle": Get nodes with shuffle. Raise `graphlearn.OutOfRangeError`
          when all the nodes are visited. Each node will be visited and only
          be visited once.
      node_from (graphlearn.NODE | graphlearn.EDGE_SRC | graphlearn.EDGE_DST):
        `graphlearn.NODE`: get node from node data, and `t` must be a node
          type.
        `graphlearn.EDGE_SRC`: get node from source node of edge data, and `t`
          must be an edge type.
        `graphlearn.EDGE_DST`: get node from destination node of edge data, and
          `t` must be an edge type.

    Return:
      A `NodeSampler` object.
    """
    sampler = strategy2op(strategy, "NodeSampler")
    return getattr(samplers, sampler)(self,
                                      t,
                                      batch_size=batch_size,
                                      strategy=strategy,
                                      node_from=node_from)

  def edge_sampler(self,
                   edge_type,
                   batch_size=64,
                   strategy="by_order"):
    """Sampler for sample one type of edges.

    Args:
      edge_type (string): Sample edges of the edge_type.
      batch_size (int, Optional): How many edges will be returned for get().
      strategy (string, Optional):Indicates how to sample edges,
        "by_order", "random" and "shuffle" are supported.
        "by_order": Get edge by order. Raise `graphlearn.OutOfRangeError` when
          all the edges are visited. Each edge will be visited and only be
          visited once.
        "random": Randomly get edges. No visting state will be hold, so out of
          range will not happened.
        "shuffle": Get edges with shuffle. Raise `graphlearn.OutOfRangeError`
          when all the edges are visited. Each edge will be visited and only
          be visited once.

    Return:
      An `EdgeSampler` object.

    """
    sampler = strategy2op(strategy, "EdgeSampler")
    return getattr(samplers, sampler)(self,
                                      edge_type,
                                      batch_size=batch_size,
                                      strategy=strategy)

  def neighbor_sampler(self,
                       meta_path,
                       expand_factor,
                       strategy="random"):
    """ Sampler for sample neighbors and edges along a meta path.
    The neighbors of the seed nodes and edges between seed nodes and
    neighbors are called layer.

    Args:
      meta_path (list): A list of edge_type.
      expand_factor (int | list): Number of neighbors sampled from all
        the dst nodes of the node along the corresponding meta_path.
        int: indicates the number of neighbors sampled for each node.
        list of int: indicates the number of neighbors sampled for each node of
          each hop, and length of expand_factor is same with length of meta_path.
      strategy (string): Indicates how to sample meta paths,
        "edge_weight", "random", "in_degree" are supported.
        "edge_weight": sample neighbor nodes by the edge weight between seed
          node and dst node.
        "random": randomly sample neighbor nodes.
        "in_degree": sample neighbor nodes by the in degree of the neighbors.

    Return:
      A 'NeighborSampler' object.
    """
    sampler = strategy2op(strategy, "NeighborSampler")
    return getattr(samplers, sampler)(self,
                                      meta_path,
                                      expand_factor,
                                      strategy=strategy)

  def negative_sampler(self,
                       object_type,
                       expand_factor,
                       strategy="random"):
    """Sampler for sample negative dst nodes of the given src nodes
    with edge_type.

    Args:
      edge_type (string): Sample negative nodes of the source node with
        specified edge_type.
      strategy (string or list): Indicates how to sample negative edges,
        "random" and "in_degree" are supported.
        "random": randomly sample negative nodes.
        "in_degree": sample negative nodes by the in degree of the target nodes.
      expand_factor (int): Indicates how many negatives to sample for one node.

    Return:
      A 'NegativeSampler' object.
    """
    sampler = strategy2op(strategy, "NegativeSampler")
    return getattr(samplers, sampler)(self,
                                      object_type,
                                      expand_factor,
                                      strategy=strategy)

  def _construct_node_source(self, path, node_type, decoder=None):
    """ Construct `NodeSource` with path, node_type and
    (attr_delimiter, data_format and attr_types) in decoder.
    """
    source = pywrap.NodeSource()
    source.id_type = node_type
    self._common_construct_source(source, path, decoder)
    return source

  def _construct_edge_source(self, path, edge_type, decoder,
                             direction=pywrap.Direction.ORIGIN):
    """ Construct `EdgeSource` with path, edge_type and
        (attr_delimiter, data_format and attr_types) in decoder.
    """
    source = pywrap.EdgeSource()
    if isinstance(edge_type, tuple) and len(edge_type) == 3:
      source.src_id_type, source.dst_id_type, source.edge_type = edge_type
    else:
      raise ValueError("edge_type param for .edge must be a tuple with "
                       "(src_type, dst_type, edge_type)")
    source.direction = direction
    self._common_construct_source(source, path, decoder)
    return source

  def _common_construct_source(self, source, path, decoder):
    """Construct pywrap.Source with decoder
    """
    source.path = path
    source.format = decoder.data_format

    if decoder.attributed:
      source.delimiter = decoder.attr_delimiter
      for t in decoder.attr_types:
        if isinstance(t, tuple):
          type_str = t[0]
          if len(t) < 2:
            source.append_hash_bucket(0)
          else:
            source.append_hash_bucket(t[1])
        else:
          type_str = t
          source.append_hash_bucket(0)

        if type_str == "int":
          source.append_attr_type(pywrap.DataType.INT64)
        elif type_str == "float":
          source.append_attr_type(pywrap.DataType.FLOAT)
        else:
          source.append_attr_type(pywrap.DataType.STRING)

  def lookup_nodes(self, node_type, ids):
    """ Get all the node properties.
    """
    req = pywrap.new_lookup_nodes_req(node_type)
    pywrap.set_lookup_nodes_req(req, ids)

    res = pywrap.new_lookup_nodes_res()
    raise_exception_on_not_ok_status(self._client.lookup_nodes(req, res))

    decoder = self.get_node_decoder(node_type)
    weights = pywrap.get_node_weights_res(res) \
      if decoder.weighted else None
    labels = pywrap.get_node_labels_res(res) \
      if decoder.labeled else None
    int_attrs = pywrap.get_node_int_attr_res(res) \
      if decoder.attributed else None
    float_attrs = pywrap.get_node_float_attr_res(res) \
      if decoder.attributed else None
    string_attrs = pywrap.get_node_string_attr_res(res) \
      if decoder.attributed else None
    int_attrs, float_attrs, string_attrs = \
      decoder.format_attrs(int_attrs, float_attrs, string_attrs)

    pywrap.del_lookup_nodes_res(res)
    pywrap.del_lookup_nodes_req(req)

    return Values(int_attrs=int_attrs,
                  float_attrs=float_attrs, string_attrs=string_attrs,
                  weights=weights, labels=labels, graph=self)

  def lookup_edges(self, edge_type, src_ids, edge_ids):
    """ Get all the edge properties.
    """
    batch_size = src_ids.flatten().size
    if batch_size != edge_ids.flatten().size:
      raise ValueError("src_ids and edge_ids for lookup edges must "
                       "be same, got {} and {}"
                       .format(batch_size, edge_ids.flatten().size))

    req = pywrap.new_lookup_edges_req(edge_type)
    pywrap.set_lookup_edges_req(req, src_ids, edge_ids)

    res = pywrap.new_lookup_edges_res()
    raise_exception_on_not_ok_status(self._client.lookup_edges(req, res))

    decoder = self.get_edge_decoder(edge_type)
    weights = pywrap.get_edge_weights_res(res) \
      if decoder.weighted else None
    labels = pywrap.get_edge_labels_res(res) \
      if decoder.labeled else None
    int_attrs = pywrap.get_edge_int_attr_res(res) \
      if decoder.attributed else None
    float_attrs = pywrap.get_edge_float_attr_res(res) \
      if decoder.attributed else None
    string_attrs = pywrap.get_edge_string_attr_res(res) \
      if decoder.attributed else None
    int_attrs, float_attrs, string_attrs = \
      decoder.format_attrs(int_attrs, float_attrs, string_attrs)

    pywrap.del_lookup_edges_res(res)
    pywrap.del_lookup_edges_req(req)

    return Values(int_attrs=int_attrs,
                  float_attrs=float_attrs, string_attrs=string_attrs,
                  weights=weights, labels=labels, graph=self)

  def get_client(self):
    return self._client
