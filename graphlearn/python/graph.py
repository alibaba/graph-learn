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
import base64
import json
import os
import warnings

from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python import sampler as samplers
from graphlearn.python.client import Client
from graphlearn.python.decoder import Decoder
from graphlearn.python.errors import raise_exception_on_not_ok_status
from graphlearn.python.query import VertexQuery, EdgeQuery, QueryEngine
from graphlearn.python.server import Server
from graphlearn.python.state import State
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

    self._server = None
    self._client = None

    self.node_state = State()
    self.edge_state = State()

    self._with_vineyard = False
    self._vineyard_handle = None

  def vineyard(self, handle, nodes=None, edges=None):
    self._with_vineyard = True
    if not isinstance(handle, dict):
      handle = json.loads(base64.b64decode(handle).decode('utf-8'))
    self._vineyard_handle = handle
    pywrap.set_vineyard_graph_id(handle['vineyard_id'])
    pywrap.set_vineyard_ipc_socket(handle['vineyard_socket'])

    for node_info in handle['node_schema']:
      confs = node_info.split(':')
      if len(confs) != 6:
        continue
      node_type = confs[0]
      if nodes is not None and node_type not in nodes:
        continue
      weighted = confs[1] == 'true'
      labeled = confs[2] == 'true'
      n_int = int(confs[3])
      n_float = int(confs[4])
      n_string = int(confs[5])
      self.node(source='',
                node_type=node_type,
                decoder=self._make_vineyard_decoder(
                  weighted, labeled, n_int, n_float, n_string))

    for edge_info in handle['edge_schema']:
      confs = edge_info.split(':')
      if len(confs) != 8:
        continue
      src_node_type = confs[0]
      edge_type = confs[1]
      dst_node_type = confs[2]
      if edges is not None and [src_node_type, edge_type, dst_node_type] not in edges \
        and (src_node_type, edge_type, dst_node_type) not in edges:
        continue
      weighted = confs[3] == 'true'
      labeled = confs[4] == 'true'
      n_int = int(confs[5])
      n_float = int(confs[6])
      n_string = int(confs[7])
      self.edge(source='',
                edge_type=(src_node_type, dst_node_type, edge_type),
                decoder=self._make_vineyard_decoder(
                  weighted, labeled, n_int, n_float, n_string))

    return self

  def _make_vineyard_decoder(self,
      weighted, labeled, n_int, n_float, n_string):
    attr_types = []
    if n_int == 0 and n_float == 0 and n_string == 0:
      attr_types = None
    else:
      attr_types.extend(["int"] * n_int)
      attr_types.extend(["float"] * n_float)
      attr_types.extend(["string"] * n_string)
    return Decoder(weighted, labeled, attr_types)

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

  def _copy_node_source(self, node):
    result = pywrap.NodeSource()
    result.path = node.path
    result.format = node.format
    result.id_type = node.id_type
    result.attr_types = node.attr_types
    result.delimiter = node.delimiter
    result.hash_buckets = node.hash_buckets
    result.ignore_invalid = node.ignore_invalid
    result.view_type = node.view_type
    result.use_attrs = node.use_attrs
    return result

  def node_view(self, node_view_type, node_type, seed=0, nsplit=1, split_range=(0, 1)):
    node_source = None
    for node in self._node_sources:
      if node.id_type == node_type:
        node_source = self._copy_node_source(node)
        break
    if node_source is None:
      raise ValueError('Node type "%s" doesn\'t exist.' % (node_type,))
    node_source.id_type = node_view_type
    node_source.view_type = '%s:%d:%d:%d:%d' % (node_type, seed, nsplit,
                                                split_range[0], split_range[1])
    self._node_decoders[node_view_type] = self._node_decoders[node_type]
    self._node_sources.append(node_source)
    return self

  def node_attributes(self, node_type, attrs, n_int, n_float, n_string):
    node_source = None
    for node in self._node_sources:
      if node.id_type == node_type:
        node_source = node
        break
    if node_source is None:
      raise ValueError('Node type "%s" doesn\'t exist.' % (node_type,))
    node_source.use_attrs = ';'.join(attrs)
    decoder = self._node_decoders[node_type]
    self._node_decoders[node_type] = self._make_vineyard_decoder(
      decoder.weighted, decoder.labeled, n_int, n_float, n_string)
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

  def _copy_edge_source(self, edge):
    result = pywrap.EdgeSource()
    result.path = edge.path
    result.format = edge.format
    result.edge_type = edge.edge_type
    result.src_id_type = edge.src_id_type
    result.dst_id_type = edge.dst_id_type
    result.attr_types = edge.attr_types
    result.delimiter = edge.delimiter
    result.hash_buckets = edge.hash_buckets
    result.ignore_invalid = edge.ignore_invalid
    result.direction = edge.direction
    result.view_type = edge.view_type
    result.use_attrs = edge.use_attrs
    return result

  def edge_attributes(self, edge_type, attrs, n_int, n_float, n_string):
    edge_source = None
    for edge in self._edge_sources:
      if edge.edge_type == edge_type:
        edge_source = edge
        break
    if edge_source is None:
      raise ValueError('edge type "%s" doesn\'t exist.' % (edge_type,))
    edge_source.use_attrs = ';'.join(attrs)
    decoder = self._edge_decoders[edge_type]
    self._edge_decoders[edge_type] = self._make_vineyard_decoder(
      decoder.weighted, decoder.labeled, n_int, n_float, n_string)
    return self

  def init(self, task_index=0, task_count=1,
           cluster="", job_name="", **kwargs):
    if self._with_vineyard:
      pywrap.set_storage_mode(8)
      pywrap.set_tracker_mode(0)

    """ Initialize the graph with creating graph server instance with
    given cluster env info.

    Args:
      task_index (int): Current task index in in_memory mode or current
        server or client index in independent mode.
      task_count (int): Total task count in in_memory mode.
      cluster (dict | josn str): Empty dict or string when Graph runs with
        local mode. Otherwise, cluster includes (server_count, client_count,
        tracker) or (server, client) or (server, client_count)
        server_count (int): count of servers.
        client_count (int): count of clients.
        tracker (str): tracker path.
        server (string): hosts of servers, split by ','.
      job_name (str): `client` or `server`, default empty means Graph runs
        with local mode.
      kwargs:
        tracker (string): tracker path for in-memory mode.
        hosts (string): hosts of servers for in-memory mode.
    """
    # In memory mode, local or distribute.
    if not job_name:
      assert cluster == ""
      tracker = kwargs.get("tracker", 'root://graphlearn')
      hosts = kwargs.get("hosts")
      host = "0.0.0.0:0"
      if hosts:
        pywrap.set_server_hosts(hosts)
        hosts = hosts.split(',')
        host = hosts[task_index]
        task_count = len(hosts)
      assert task_index < task_count

      # Local in-memory mode.
      if task_count == 1:
        pywrap.set_deploy_mode(0)
      # Distribute in-memory mode.
      else:
        pywrap.set_deploy_mode(2)
        pywrap.set_server_count(task_count)
        pywrap.set_client_count(task_count)
        pywrap.set_tracker(tracker)
        pywrap.set_client_id(task_index)

      self._client = Client(client_id=task_index)
      self._server = Server(task_index, task_count, host, tracker)
      self._server.start()
      self._server.init(self._edge_sources, self._node_sources)

    # Distribute service mode.
    else:
      if isinstance(cluster, dict):
        cluster_spec = cluster
      elif isinstance(cluster, str):
        cluster_spec = json.loads(cluster)
      else:
        raise ValueError("cluster must be dict or json string.")

      tracker = cluster_spec.get("tracker", 'root://graphlearn')
      server_count = cluster_spec.get("server_count")
      servers = cluster_spec.get("server")
      if servers:
        pywrap.set_server_hosts(servers)
        servers = servers.split(',')
        server_count = len(servers)

      client_count = cluster_spec.get("client_count")
      clients = cluster_spec.get("client")
      if clients:
        client_count = len(clients.split(','))
      if not server_count or not client_count:
        raise ValueError("cluster is composed of"
                         " (server_count, client_count, tracker)"
                         " or (server, client) or (server, client_count)}")
      pywrap.set_server_count(server_count)
      pywrap.set_client_count(client_count)
      pywrap.set_deploy_mode(1)

      if job_name == "client":
        pywrap.set_tracker(tracker)
        pywrap.set_client_id(task_index)
        self._client = Client(client_id=task_index, in_memory=False)
        self._server = None
      elif job_name == "server":
        self._client = None
        server_host = "0.0.0.0:0" if not servers else servers[task_index]
        self._server = Server(task_index, server_count, server_host, tracker)
        self._server.start()
        self._server.init(self._edge_sources, self._node_sources)
      else:
        raise ValueError("Only support client and server for GL.")

    return self

  def init_vineyard(self, server_index=None, worker_index=None, worker_count=None,
                    standalone=False):
    if not self._with_vineyard:
      raise ValueError('Not a vineyard graph')

    if standalone:
      self.init()
      return self

    if server_index is None and worker_index is None:
      raise ValueError('Cannot decide to launch a server or a worker')
    if server_index is not None and worker_index is not None:
      raise ValueError('Cannot be a server and a worker at the same unless standalone is True')

    cluster = {'server': self._vineyard_handle['server'],
               'client_count': self._vineyard_handle['client_count']}
    if server_index is not None:
      self.init(cluster=cluster, task_index=server_index, job_name="server")
    else:
      self.init(cluster=cluster, task_index=worker_index, job_name="client")
    return self

  def close(self):
    self.wait_for_close()

  def wait_for_close(self):
    if self._client:
      self._client.stop()
      self._client = None
    if self._server:
      self._server.stop()
      self._server = None

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
    req = pywrap.new_lookup_nodes_request(node_type)
    pywrap.set_lookup_nodes_request(req, ids)

    res = pywrap.new_lookup_nodes_response()
    status = self._client.lookup_nodes(req, res)
    if status.ok():
      decoder = self.get_node_decoder(node_type)
      weights = weights = pywrap.get_node_weights(res) \
        if decoder.weighted else None
      labels = pywrap.get_node_labels(res) \
        if decoder.labeled else None
      int_attrs = pywrap.get_node_int_attributes(res) \
        if decoder.attributed else None
      float_attrs = pywrap.get_node_float_attributes(res) \
        if decoder.attributed else None
      string_attrs = pywrap.get_node_string_attributes(res) \
        if decoder.attributed else None
      int_attrs, float_attrs, string_attrs = \
        decoder.format_attrs(int_attrs, float_attrs, string_attrs)

    pywrap.del_op_response(res)
    pywrap.del_op_request(req)
    raise_exception_on_not_ok_status(status)

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

    req = pywrap.new_lookup_edges_request(edge_type)
    pywrap.set_lookup_edges_request(req, src_ids, edge_ids)

    res = pywrap.new_lookup_edges_response()
    status = self._client.lookup_edges(req, res)
    if status.ok():
      decoder = self.get_edge_decoder(edge_type)
      weights = pywrap.get_edge_weights(res) \
        if decoder.weighted else None
      labels = pywrap.get_edge_labels(res) \
        if decoder.labeled else None
      int_attrs = pywrap.get_edge_int_attributes(res) \
        if decoder.attributed else None
      float_attrs = pywrap.get_edge_float_attributes(res) \
        if decoder.attributed else None
      string_attrs = pywrap.get_edge_string_attributes(res) \
        if decoder.attributed else None
      int_attrs, float_attrs, string_attrs = \
        decoder.format_attrs(int_attrs, float_attrs, string_attrs)

    pywrap.del_op_response(res)
    pywrap.del_op_request(req)
    raise_exception_on_not_ok_status(status)

    return Values(int_attrs=int_attrs,
                  float_attrs=float_attrs, string_attrs=string_attrs,
                  weights=weights, labels=labels, graph=self)

  def get_client(self):
    return self._client
