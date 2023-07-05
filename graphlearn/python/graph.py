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

import atexit
import base64
import json
import sys
import warnings

import numpy as np

from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python.client import Client
from graphlearn.python.server import Server
import graphlearn.python.data as data
import graphlearn.python.errors as errors
import graphlearn.python.gsl as gsl
import graphlearn.python.operator as ops
import graphlearn.python.sampler as samplers
import graphlearn.python.utils as utils
from graphlearn.python.sampler.subgraph_sampler import SubGraphSampler

class Graph(object):
  """ Entry of graph data operations, such as cluster initialization, data
  loading and sampling. Both homogeneous and heterogeneous graphs are supported.

  To use GL, we should define a Graph object first.
  ```
  import graphlearn as gl
  g = gl.Graph()
  ```

  Based on a Graph object, we can do things through the APIs below.

  node(): Add a kind of VERTEX with vertex type and data schema.
  edge(): Add a kind of EDGE with (src_type, edge_type, dst_type) tuple and data schema.
  init(): Load and initialize graph data. Cluster info is needed in distributed mode.
  get_topology(): Get topology info for heterogeneous graph.
  node_sampler(): Create a VERTEX sampler to iterate vertices of given type in this graph.
  edge_sampler(): Create an EDGE sampler to iterate edges of given type in this graph.
  neighbor_sampler(): Create a NEIGHBOR sampler to sample neighbors for given
    vertices according to a metapath.
  negative_sampler(): Create a NEGATIVE sampler to sample un-neighbored vertices
    for given inputs according to a metapath.
  subgraph_sampler(): Create SUBGRAPH sampler to sample a sub-graph.
  V(): Entry of GSL, starting with VERTEX sampling.
  E(): Entry of GSL, starting with EDGE sampling.
  """

  def __init__(self):
    # list of NodeSource added by .node()
    self._node_sources = []
    # list of EdgeSource added by .edge()
    self._edge_sources = []
    # list of NodeSource added by .node()

    # maintain the graph's static topology for fast query.
    self._topology = data.Topology()
    # maintain a map of node_type with it's decoder
    self._node_decoders = {}
    # maintain a map of edge_type with it's decoder
    self._edge_decoders = {}
    self._undirected_edges = []

    self._server = None
    self._client = None

    self.node_state = data.NodeState()
    self.edge_state = data.EdgeState()

    self._datasets = []

    self._with_vineyard = False
    self._vineyard_handle = None

    def stop_sampling():
      if self._server:
        self._server.stop_sampling()
    atexit.register(stop_sampling)

  def vineyard(self, handle, nodes=None, edges=None):
    """ Initialize a graph from vineyard.

    :code:`handle` is the schema information of the vineyard graph, as shown in
    the following example:

    .. code:: python

        {
            "server": "127.0.0.1:8888,127.0.0.1:8889",
            "client_count": 1,
            "vineyard_socket": "/var/run/vineyard.sock",
            "vineyard_id": 13278328736,
            "fragments": [13278328736, ...],  # fragment ids, may be None
            "node_schema": [
                "user:false:false:10:0:0",
                "item:true:false:0:0:5"
            ],
            "edge_schema": [
                "user:click:item:true:false:0:0:0",
                "user:buy:item:true:true:0:0:0",
                "item:similar:item:false:false:10:0:0"
            ],
            "node_attribute_types": {
                "person": {
                    "age": "i",
                    "name": "s",
                },
            },
            "edge_attribute_types": {
                "knows": {
                    "weight": "f",
                },
            },
        }

    Fields in :code:`schema` are:

      + the name of node type or edge type
      + whether the graph is weighted graph
      + whether the graph is labeled graph
      + the number of int attributes
      + the number of float attributes
      + the number of string attributes
    """
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
                  weighted, labeled, False, n_int, n_float, n_string))

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
      if 'reverse' not in edge_type:
        self._undirected_edges.append(edge_type)
      weighted = confs[3] == 'true'
      labeled = confs[4] == 'true'
      n_int = int(confs[5])
      n_float = int(confs[6])
      n_string = int(confs[7])
      self.edge(source='',
                edge_type=(src_node_type, dst_node_type, edge_type),
                decoder=self._make_vineyard_decoder(
                  weighted, labeled, False, n_int, n_float, n_string))

    return self

  def _make_vineyard_decoder(self,
      weighted, labeled, timestamped, n_int, n_float, n_string):
    attr_types = []
    if n_int == 0 and n_float == 0 and n_string == 0:
      attr_types = None
    else:
      attr_types.extend(["int"] * n_int)
      attr_types.extend(["float"] * n_float)
      attr_types.extend(["string"] * n_string)
    return data.Decoder(weighted, labeled, timestamped, attr_types)

  def node(self,
           source,
           node_type,
           decoder,
           option=None,
           mask=utils.Mask.NONE):
    """ Add graph vertices that will be loaded from a given path.

    Args:
      source (string): Data source path where to load the nodes.
      node_type (string): Indicates the type of the added nodes.
      decoder (Decoder): A Decoder object to describe the data schema.
      mask (TRAIN | TEST | VAL): Mark the source as TRAIN data, TEST data
        or VAL data for the given node_type in the graph.
    """
    if not isinstance(source, str):
      raise ValueError('source for node() must be string.')
    if not isinstance(node_type, str):
      raise ValueError('node_type for node() must be string.')
    if not isinstance(decoder, data.Decoder):
      raise ValueError('decoder must be an instance of `Decoder`, got {}'
                       .format(type(decoder)))

    node_type = utils.get_mask_type(node_type, mask)
    self._node_decoders[node_type] = decoder
    sources = [x.strip() for x in source.split(',')]
    for src_file in sources:
      node_source = self._construct_node_source(
          src_file, node_type, decoder, option)
      self._node_sources.append(node_source)
    return self

  def _copy_node_source(self, node):
    result = pywrap.NodeSource()
    result.path = node.path
    result.id_type = node.id_type
    result.format = node.format
    result.attr_info = node.attr_info
    result.option = node.option
    result.view_type = node.view_type
    result.use_attrs = node.use_attrs
    return result

  def node_view(self, node_type, mask=utils.Mask.NONE, seed=0, nsplit=1, split_range=(0, 1)):
    """ Add a virtual view of given `node_type` to split nodes into train/val/test set
        without duplicately loading the graph.

        It is for vineyard backend only.
    """
    assert self._with_vineyard, "node_view() is only for vineyard backend"
    node_source = None
    for node in self._node_sources:
      if node.id_type == node_type:
        node_source = self._copy_node_source(node)
        break
    if node_source is None:
      raise ValueError('Node type "%s" doesn\'t exist.' % (node_type,))
    masked_node_type = utils.get_mask_type(node_type, mask)
    node_source.id_type = masked_node_type
    node_source.view_type = '%s:%d:%d:%d:%d' % (node_type, seed, nsplit,
                                                split_range[0], split_range[1])
    self._node_decoders[masked_node_type] = self._node_decoders[node_type]
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
      decoder.weighted, decoder.labeled, decoder.timestamped, n_int, n_float, n_string)
    return self

  def edge(self,
           source,
           edge_type,
           decoder=None,
           directed=True,
           option=None,
           mask=utils.Mask.NONE):
    """ Add graph edges that will be loaded from a given path.

    Args:
      source (string): Data source path where to load the edges.
      edge_type (tuple): A tuple of (src_type, dst_type, edge_type) that
        indicates types of the edges.
      decoder (Decoder): A Decoder object to describe the data schema.
      directed (boolean): Whether edges are directed.
      mask (TRAIN | TEST | VAL): Mark the source as TRAIN data, TEST data
        or VAL data for the given edge_type in the graph.
    """
    if not isinstance(source, str):
      raise ValueError('source for edge() must be a string.')
    if not isinstance(edge_type, tuple) or len(edge_type) != 3:
      raise ValueError("edge_type for edge() must be a tuple of "
                       "(src_type, dst_tye, edge_type).")
    if not decoder:
      decoder = data.Decoder()
    if not isinstance(decoder, data.Decoder):
      raise ValueError('decoder must be an instance of Decoder, got {}'
                       .format(type(decoder)))

    masked_edge_type = utils.get_mask_type(edge_type[2], mask)
    self._edge_decoders[masked_edge_type] = decoder

    self._topology.add(masked_edge_type, edge_type[0], edge_type[1])
    sources = [x.strip() for x in source.split(',')]
    for src_file in sources:
      edge_source = self._construct_edge_source(
          src_file, (edge_type[0], edge_type[1], masked_edge_type),
          decoder,
          direction=pywrap.Direction.ORIGIN,
          option=option)
      self._edge_sources.append(edge_source)

    if not directed:
      self.add_reverse_edges(edge_type, source, decoder, option)
    return self

  def _copy_edge_source(self, edge):
    result = pywrap.EdgeSource()
    result.path = edge.path
    result.edge_type = edge.edge_type
    result.src_id_type = edge.src_id_type
    result.dst_id_type = edge.dst_id_type
    result.format = edge.format
    result.direction = edge.direction
    result.attr_info = edge.attr_info
    result.option = edge.option
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
      decoder.weighted, decoder.labeled, decoder.timestamped, n_int, n_float, n_string)
    return self

  @property
  def undirected_edges(self):
    return self._undirected_edges

  def add_reverse_edges(self, edge_type, source, decoder, option):
    self._undirected_edges.append(edge_type[2])
    sources = [x.strip() for x in source.split(',')]

    if (edge_type[0] != edge_type[1]):  # pylint: disable=superfluous-parens
      reversed_edge_type = edge_type[2] + '_reverse'
      self._edge_decoders[reversed_edge_type] = decoder
      self._topology.add(reversed_edge_type, edge_type[1], edge_type[0])
      for src_file in sources:
        edge_source_reverse = self._construct_edge_source(
            src_file,
            (edge_type[1], edge_type[0], reversed_edge_type),
            decoder,
            direction=pywrap.Direction.REVERSED,
            option=option)
        self._edge_sources.append(edge_source_reverse)
    else:
      for src_file in sources:
        edge_source_reverse = self._construct_edge_source(
            src_file,
            edge_type,
            decoder,
            direction=pywrap.Direction.REVERSED,
            option=option)
        self._edge_sources.append(edge_source_reverse)
  def init(self, task_index=0, task_count=1,
           cluster="", job_name="", **kwargs):
    """ Initialize the graph object in local mode or distributed mode.

    If deployed in local mode, just call `g.init()` without any parameters.
    If in distributed, you should care about WORKER mode and SERVER mode.

    Args:
      task_index (int): Current task index.
        If in WORKER mode, it means the current worker index.
        If in SERVER mode, it means the current server index.
      task_count (int): Total task count. Only needed in WORKER mode.
        It means in the total worker count.
      cluster (dict | json string): Only needed in SERVER mode.
        3 kinds of schemas are supported:
        cluster = {
          "server_count": 2,
          "client_count": 4,
          "tracker": "root://graphlearn"
        }
        cluster = {
          "server": "127.0.0.2:6666,127.0.0.3:7777",
          "client": "127.0.0.4:8888,127.0.0.5:9999"
        }
        cluster = {
          "server": "127.0.0.2:6666,127.0.0.3:7777",
          "client_count", 2
        }
        server_count (int): count of servers.
        client_count (int): count of clients.
        server (string): hosts of servers, split by ','.
      job_name (str): `client` or `server`. Only needed in SERVER mode.
      kwargs:
        tracker (string): Optional tracker path for WORKER mode.
        hosts (string): Optional worker hosts for WORKER mode.
    """
    if self._with_vineyard:
      pywrap.set_storage_mode(8)
      pywrap.set_tracker_mode(0)

    if not cluster and task_count == 1:
      # Local mode
      pywrap.set_deploy_mode(pywrap.DeployMode.LOCAL)
      self.deploy_in_local_mode(task_index)
    elif not cluster:
      if task_count > 1 or kwargs.get("hosts") is not None:
        # WORKER mode
        pywrap.set_deploy_mode(pywrap.DeployMode.WORKER)
        tracker = kwargs.get("tracker", "root://graphlearn")
        hosts = kwargs.get("hosts")
        self.deploy_in_worker_mode(tracker, hosts, task_index, task_count)
    else:
      # SERVER mode
      pywrap.set_deploy_mode(pywrap.DeployMode.SERVER)
      self.deploy_in_server_mode(task_index, cluster, job_name)
    return self

  def deploy_in_local_mode(self, task_index):
    self._client = Client(client_id=task_index)
    self._server = Server(0, 1, "", "")
    self._server.start()
    self._server.init(self._edge_sources, self._node_sources)

  def deploy_in_worker_mode(self, tracker, hosts, task_index, task_count):
    if hosts:
      pywrap.set_server_hosts(hosts)
      hosts = hosts.split(',')
      host = hosts[task_index]
      task_count = len(hosts)
    else:
      host = "0.0.0.0:0"

    assert task_index < task_count

    pywrap.set_client_id(task_index)
    pywrap.set_client_count(task_count)
    pywrap.set_server_count(task_count)
    pywrap.set_tracker(tracker)

    self._client = Client(client_id=task_index)
    self._server = Server(task_index, task_count, host, tracker)
    self._server.start()
    self._server.init(self._edge_sources, self._node_sources)

  def deploy_in_server_mode(self, task_index, cluster, job_name):
    if isinstance(cluster, dict):
      cluster_spec = cluster
    elif isinstance(cluster, str):
      cluster_spec = json.loads(cluster)
    else:
      raise ValueError("cluster must be dict or json string.")

    tracker = cluster_spec.get("tracker", "root://graphlearn")

    # parse servers
    server_count = cluster_spec.get("server_count")
    servers = cluster_spec.get("server")
    if servers:
      pywrap.set_server_hosts(servers)
      servers = servers.split(',')
      server_count = len(servers)

    # parse clients
    client_count = cluster_spec.get("client_count")
    clients = cluster_spec.get("client")
    if clients:
      client_count = len(clients.split(','))

    if not server_count or not client_count:
      raise ValueError("Invalid cluster schema")

    pywrap.set_server_count(server_count)
    pywrap.set_client_count(client_count)

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
      raise ValueError("Only support client and server job name in SERVER mode.")

  def add_dataset(self, ds):
    self._datasets.append(ds)

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
      # re-target to fragment id, if exists
      if self._vineyard_handle.get('fragments', None):
        pywrap.set_vineyard_graph_id(self._vineyard_handle['fragments'][server_index])
      self.init(cluster=cluster, task_index=server_index, job_name="server")
    else:
      self.init(cluster=cluster, task_index=worker_index, job_name="client")
    return self

  def close(self):
    self.wait_for_close()

  def wait_for_close(self):
    for ds in self._datasets:
      ds.close()
    if self._client:
      self._client.stop()
      self._client = None
    if self._server:
      self._server.stop()
      self._server = None

  def V(self,
        t,
        feed=None,
        node_from=pywrap.NodeFrom.NODE,
        mask=utils.Mask.NONE):
    """ Entry of GSL, starting from VERTEX.

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
      mask (NONE | TRAIN | TEST | VAL): The given node set is indexed by both the
        raw node type and mask value. The default mask value is NONE, which plays
        nothing on the index.
    """
    if feed is not None:
      raise NotImplementedError("`feed` is not supported for now.")
    dag = gsl.Dag(self)
    params = {pywrap.kNodeType: utils.get_mask_type(t, mask),
              pywrap.kNodeFrom: int(node_from)}
    source_node = gsl.TraverseVertexDagNode(
      dag, op_name="GetNodes", params=params)
    source_node.set_output_field(pywrap.kNodeIds)
    source_node.set_path(t, node_from)
    dag.root = source_node

    # Add sink node to dag
    gsl.SinkNode(dag)
    return source_node

  def E(self,
        edge_type,
        feed=None,
        reverse=False):
    """ Entry of GSL, starting from EDGE.

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
    """
    if feed is not None:
      raise NotImplementedError("`feed` is not supported for now.")
    dag = gsl.Dag(self)
    if reverse:
      edge_type = edge_type + "_reverse"

    dag = gsl.Dag(self)
    params = {pywrap.kEdgeType: edge_type}
    source_node = gsl.TraverseSourceEdgeDagNode(
      dag, op_name="GetEdges", params=params)
    source_node.set_output_field(pywrap.kEdgeIds)
    source_node.set_path(edge_type, pywrap.NodeFrom.NODE)
    dag.root = source_node

    # Add sink node to dag
    gsl.SinkNode(dag)
    return source_node

  def SubGraph(self,
               seed_type,
               nbr_type,
               batch_size=64,
               strategy="random_node",
               num_nbrs=[0],
               feed=None):
    """ SubGraph Sampling in GSL.
 
    Args:
      seed_type (string): Sample seed type, either node type or edge type.
      nbr_type (string): Neighbor type of seeds nodes/edges.
      batch_size (int): How many nodes will be returned each iter.
      strategy (string, Optional): Sampling strategy. "random_node/edge" and
        "in_order_node/edge" are supported.
      num_nbrs (int, Optional): number of neighbors for each hop.
    """
    if feed is not None:
      raise NotImplementedError("`feed` is not supported for now.")
    dag = gsl.Dag(self)
    assert strategy in \
      ["random_node", "random_edge", "in_order_node", "in_order_edge"]
    op_name = utils.strategy2op(strategy, "SubGraphSampler")
    params = {pywrap.kSeedType: seed_type,
              pywrap.kNbrType: nbr_type,
              pywrap.kBatchSize: batch_size,
              pywrap.kEpoch: sys.maxsize >> 32,
              pywrap.kStrategy: op_name,
              pywrap.kNeighborCount: num_nbrs}
    source_node = gsl.SubGraphDagNode(
      dag, op_name=op_name, params=params)
    source_node.set_output_field(pywrap.kNodeIds)
    source_node.set_path(nbr_type, pywrap.NodeFrom.EDGE_SRC) #homo graph.
 
    dag.root = source_node
 
    # Add sink node to dag
    gsl.SinkNode(dag)
    return source_node

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
      decoder = data.Decoder()
    return decoder

  def get_edge_decoder(self, edge_type):
    """ Get decoder of the specific edge_type.
    """
    decoder = self._edge_decoders.get(edge_type)
    if not decoder:
      warnings.warn("Edge_type {} not exist in graph. Use default decoder."
                    .format(edge_type))
      decoder = data.Decoder()
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
      ids (numpy.ndarray): ids of nodes. In sparse case, it must be 1D.
      offsets: (list): To get `SparseNodes`, whose dense
        shape is 2D, `offsets` indicates the number of nodes for each line.
        Default None means it is a dense `Nodes`.
      shape (tuple, Optional): Indicates the shape of nodes ids, attrs, etc.
        For dense case, default None means ids.shape. For sparse case, it
        must has a value which indicates the 2D dense shape.

    Return:
      A `Nodes` object or a `SparseNodes` object.
    '''
    if offsets is None:
      nodes = data.Nodes(ids, node_type, shape=shape, graph=self)
    else:
      # Specially, for case where there is no dense shape,
      # such as sampling full neighbors,
      # reset the shape as [batch_size, max(neighbor_counts)].
      shape = (len(offsets), shape[1] if shape[1] and shape[1] > 0 else max(offsets))
      nodes = data.SparseNodes(ids, offsets, shape, node_type, graph=self)
    return nodes

  def get_edges(self, edge_type, src_ids, dst_ids, edge_ids=None, offsets=None,
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

    src_type = self._topology.get_src_type(edge_type)
    dst_type = self._topology.get_dst_type(edge_type)
    if offsets is None:
      edges = data.Edges(src_ids, src_type,
                         dst_ids, dst_type,
                         edge_type,
                         edge_ids,
                         shape=shape,
                         graph=self)
    else:
      shape = (len(offsets), shape[1] if shape[1] and shape[1] > 0 else max(offsets))
      edges = data.SparseEdges(src_ids, src_type,
                               dst_ids, dst_type,
                               edge_type,
                               offsets, shape,
                               edge_ids,
                               graph=self)
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
                   node_from=pywrap.NodeFrom.NODE,
                   mask=utils.Mask.NONE):
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
    sampler = utils.strategy2op(strategy, "NodeSampler")
    return getattr(samplers, sampler)(self,
                                      t,
                                      batch_size=batch_size,
                                      strategy=strategy,
                                      node_from=node_from,
                                      mask=mask)

  def edge_sampler(self,
                   edge_type,
                   batch_size=64,
                   strategy="by_order",
                   mask=utils.Mask.NONE):
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
    sampler = utils.strategy2op(strategy, "EdgeSampler")
    return getattr(samplers, sampler)(self,
                                      edge_type,
                                      batch_size=batch_size,
                                      strategy=strategy,
                                      mask=mask)

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
    sampler = utils.strategy2op(strategy, "NeighborSampler")
    return getattr(samplers, sampler)(self,
                                      meta_path,
                                      expand_factor,
                                      strategy=strategy)

  def negative_sampler(self,
                       object_type,
                       expand_factor,
                       strategy="random",
                       conditional=False,
                       **kwargs):
    """Sampler for sampling negative dst nodes of the given src nodes
    with edge_type.

    Args:
      edge_type (string): Sample negative nodes of the source node with
        specified edge_type.
      strategy (string or list): Indicates how to sample negative edges,
        "random" and "in_degree" are supported.
        "random": randomly sample negative nodes.
        "in_degree": sample negative nodes by the in degree of the target nodes.
      expand_factor (int): Indicates how many negatives to sample for one node.
      conditional(bool): Indicates whether sample under condition.
    Return:
      A 'NegativeSampler' object.
    """
    if not conditional:
      sampler = utils.strategy2op(strategy, "NegativeSampler")
      return getattr(samplers, sampler)(self,
                                        object_type,
                                        expand_factor,
                                        strategy=strategy)
    else:
      return getattr(samplers, "ConditionalNegativeSampler")(self,
                                                             object_type,
                                                             expand_factor,
                                                             strategy=strategy,
                                                             **kwargs)

  def _construct_node_source(self, path, node_type, decoder=None, option=None):
    source = pywrap.NodeSource()
    source.id_type = node_type
    self._common_construct_source(source, path, decoder, option)
    return source

  def _construct_edge_source(self,
                             path,
                             edge_type,
                             decoder,
                             direction=pywrap.Direction.ORIGIN,
                             option=None):
    source = pywrap.EdgeSource()
    if isinstance(edge_type, tuple) and len(edge_type) == 3:
      source.src_id_type, source.dst_id_type, source.edge_type = edge_type
    else:
      raise ValueError("edge_type param for .edge must be a tuple with "
                       "(src_type, dst_type, edge_type)")
    source.direction = direction
    self._common_construct_source(source, path, decoder, option)
    return source

  def _common_construct_source(self, source, path, decoder, option=None):
    """Construct pywrap.Source
    """
    source.path = path
    source.format = decoder.data_format
    if option:
      source.option = option
    else:
      source.option = pywrap.IndexOption()
      source.option.name = "sort"

    if decoder.attributed:
      source.attr_info.delimiter = decoder.attr_delimiter
      for t in decoder.attr_types:
        type_str, bucket_size, is_multival = decoder.parse(t)
        if is_multival:
          source.attr_info.append_hash_bucket(0)
        elif bucket_size is not None:
          source.attr_info.append_hash_bucket(bucket_size)
        else:
          source.attr_info.append_hash_bucket(0)

        if type_str == "int":
          source.attr_info.append_type(pywrap.DataType.INT64)
        elif type_str == "float":
          source.attr_info.append_type(pywrap.DataType.FLOAT)
        else:
          source.attr_info.append_type(pywrap.DataType.STRING)

  def lookup_nodes(self, node_type, ids):
    """ Get the attributes of given nodes.
    """
    ids = np.array(ids)
    req = pywrap.new_lookup_nodes_request(node_type)
    pywrap.set_lookup_nodes_request(req, ids)

    res = pywrap.new_lookup_nodes_response()
    status = self._client.lookup_nodes(req, res)
    if status.ok():
      decoder = self.get_node_decoder(node_type)
      weights = pywrap.get_node_weights(res) if decoder.weighted else None
      labels = pywrap.get_node_labels(res) if decoder.labeled else None
      int_attrs = pywrap.get_node_int_attributes(
        res) if decoder.attributed else None
      float_attrs = pywrap.get_node_float_attributes(
        res) if decoder.attributed else None
      string_attrs = pywrap.get_node_string_attributes(
        res) if decoder.attributed else None
      int_attrs, float_attrs, string_attrs = decoder.format_attrs(
        int_attrs, float_attrs, string_attrs)

    pywrap.del_op_response(res)
    pywrap.del_op_request(req)
    errors.raise_exception_on_not_ok_status(status)

    return data.Values(
        int_attrs=int_attrs,
        float_attrs=float_attrs,
        string_attrs=string_attrs,
        weights=weights,
        labels=labels,
        graph=self)

  def lookup_edges(self, edge_type, src_ids, edge_ids):
    """ Get the attributes of given edges.
    """
    src_ids = np.array(src_ids)
    edge_ids = np.array(edge_ids)

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
      weights = pywrap.get_edge_weights(res) if decoder.weighted else None
      labels = pywrap.get_edge_labels(res) if decoder.labeled else None
      int_attrs = pywrap.get_edge_int_attributes(
        res) if decoder.attributed else None
      float_attrs = pywrap.get_edge_float_attributes(
        res) if decoder.attributed else None
      string_attrs = pywrap.get_edge_string_attributes(
        res) if decoder.attributed else None
      int_attrs, float_attrs, string_attrs = decoder.format_attrs(
        int_attrs, float_attrs, string_attrs)

    pywrap.del_op_response(res)
    pywrap.del_op_request(req)
    errors.raise_exception_on_not_ok_status(status)

    return data.Values(
        int_attrs=int_attrs,
        float_attrs=float_attrs,
        string_attrs=string_attrs,
        weights=weights,
        labels=labels,
        graph=self)

  def search(self, node_type, inputs, option):
    knn = ops.KnnOperator(self._client)
    inputs = np.array(inputs)
    return knn.search(node_type, inputs, option.k)

  def subgraph_sampler(self,
                       seed_type,
                       nbr_type,
                       num_nbrs=[0],
                       need_dist=False):
    """Sampler for sample SubGraph.

    Args:
      graph (`Graph` object): The graph which sample from.
      nbr_type (string): Neighbor type of seeds nodes/edges.
      num_nbrs (int, Optional): number of neighbors for each hop.
      need_dist: Whether need return the distance from each node in subgraph
        to src and dst. Note that this arg is valid only when `dst_ids` in
        `get()` is not None and size of `dst_ids` is 1.
    Return:
      An `SubGraphSampler` object.

    """
    return SubGraphSampler(self,
                           seed_type,
                           nbr_type,
                           num_nbrs=num_nbrs,
                           need_dist=need_dist)

  def get_stats(self):
    req = pywrap.new_get_stats_request()
    res = pywrap.new_get_stats_response()
    status = self._client.get_stats(req, res)
    stats = None
    if status.ok():
      stats = pywrap.get_stats(res)
    pywrap.del_op_response(res)
    pywrap.del_op_request(req)
    errors.raise_exception_on_not_ok_status(status)
    return stats

  def server_get_stats(self):
    return self._server.get_stats()

  def _get_degree(self, edge_type, node_from, ids):
    req = pywrap.new_get_degree_request(edge_type, node_from)
    pywrap.set_degree_request(req, ids)
    res = pywrap.new_get_degree_response()
    status = self._client.get_degree(req, res)
    if status.ok():
      degrees = pywrap.get_degree(res)
    pywrap.del_op_response(res)
    pywrap.del_op_request(req)
    errors.raise_exception_on_not_ok_status(status)
    return degrees

  def in_degrees(self, ids, edge_type):
    ids = np.array(ids)
    return self._get_degree(edge_type, pywrap.NodeFrom.EDGE_DST, ids)

  def out_degrees(self, ids, edge_type):
    ids = np.array(ids)
    return self._get_degree(edge_type, pywrap.NodeFrom.EDGE_SRC, ids)

  def get_client(self):
    return self._client
