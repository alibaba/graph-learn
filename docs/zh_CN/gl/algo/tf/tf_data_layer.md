## 数据层
数据层的一个主要目标是提供GraphLearn图操作返回的numpy数据流到TensorFlow1.x里的tensor数据流的转换。我们使用了`tf.data.Dataset`的`from_generator`接口来完成这样转换过程。此外，为了方便对节点、边特征的统一处理，我们提供了特征处理接口来完成对原始连续特征、离散特征、多值离散特征的处理，将这些特征处理成连续的向量，作为后续模型的输入。下面我们按照数据处理的先后顺序描述整个数据层的构建过程：
**基础数据对象**，描述模型层的基本数据对象。**Tensor数据对象**，描述TensorFlow tensor格式的数据格式。
**特征处理**，描述对不同类型原始特征的处理的接口。
​

### 基础数据对象
对应nn/data.py, nn/subgraph.py, nn/dataset.py


GraphLearn每个图操作的返回结果为numpy ndarray格式的`Nodes`或者`Edges`对象，为了方便模型层的处理，我们使用`Data`对象来统一表示`Nodes`和`Edges`。这样一个GSL的查询结果，可以用一个`Data` dict来表示，key是alias, value是具体的`Data`对象。


#### Data

```python
class Data(object):
  """A plain object modeling a batch of `Nodes` or `Edges`."""
  def __init__(self,
               ids=None,
               ints=None,
               floats=None,
               strings=None,
               labels=None,
               weights=None,
               **kwargs):
    """ ints, floats and strings are attributes in numpy or Tensor format 
      with the shape
      [batch_size, int_attr_num],
      [batch_size, float_attr_num],
      [batch_size, string_attr_num].
      labels and weights are in numpy or Tensor format with the shape 
      [batch_size], [batch_size].
      The data object can be extented by any other additional data.
    """
    self.ids = ids
    self.int_attrs = ints
    self.float_attrs = floats
    self.string_attrs = strings
    self.labels = labels
    self.weights = weights
    for key, item in kwargs.items():
      self[key] = item

    self._handler_dict = {}

  def apply(self, func):
    """Applies the function `func` to all attributes.
    """
    for k, v in self.__dict__.items():
      if v is not None and k[:2] != '__' and k[-2:] != '__':
        self.__dict__[k] = func(v)
    return self
```
​

​

#### SubGraph
为了方便GNNs算法的建模，我们使用`SubGraph`来表示一个采样的子图，它由`edge_index`, `nodes`和`edges`构成。由于目前GSL没有提供直接采样子图的功能，因此生成SubGraph时需要显示地使用一个`inducer`。
```python
class SubGraph(object):
  """ `SubGraph` is a basic data structure used to describe a sampled 
  subgraph. It constists of `edge_index` and nodes `Data` and edges `Data`.

  Args:
    edge_index: A np.ndarray object with shape [2, batch_size], which indicates
      [rows, cols] of SubGraph.
    nodes: A `Data` object denoting the input nodes.
    edges: A `Data` object denoting the input edges.
    
  Note that this object can be extented by any other additional data.
  """
  def __init__(self, edge_index, nodes, edges=None, **kwargs):
    self._edge_index = edge_index
    self._nodes = nodes
    self._edges = edges
    for key, item in kwargs.items():
      self[key] = item

  @property
  def num_nodes(self):
    return self._nodes.ids.size

  @property
  def num_edges(self):
    return self._edge_index.shape[1]

  @property
  def nodes(self):
    return self._nodes

  @property
  def edge_index(self):
    return self._edge_index

  @property
  def edges(self):
    return self._edges

  def __getitem__(self, key):
    return getattr(self, key, None)

  def __setitem__(self, key, value):
    setattr(self, key, value)
```

#### HeteroSubGraph（since v1.1.0）
```python
class HeteroSubGraph(object):
  """ A data object describing heterogeneous `SubGraph`.
  Different types of nodes and edges are represented by a dict.

  Args:
    edge_index_dict: A dict of np.ndarray objects. Each key is a tuple of 
      (src_type, edge_type, dst_type) and each value indicates [rows, cols].
    nodes_dict: A dict of `Data`/ndarray object denoting different types of nodes.
    edges_dict: A dict of `Data`/ndarray object denoting different types of edges.
  
  Examples:
    For meta-path "user-click-item", the HeteroSubGraph may be created as follows:
      edge_index_dict[('user', 'click', 'item')] = np.array([[0,1,2], [2,3,4]])
      edges_dict[('user', 'click', 'item')] = Data(...)
      nodes_dict['user'] = Data(...)
      nodes_dict['item'] = Data(...)
      hg = HeteroSubGraph(edge_index_dict, nodes_dict, edges_dict)

  """
  def __init__(self, edge_index_dict, nodes_dict, edges_dict=None, **kwargs):
    self._edge_index_dict = edge_index_dict
    self._nodes_dict = nodes_dict
    self._edges_dict = edges_dict
    for key, item in kwargs.items():
      self[key] = item

  def num_nodes(self, node_type):
    if isinstance(self._nodes_dict[node_type], Data):
      return self._nodes_dict[node_type].ids.size
    else:
      return self._nodes_dict[node_type].size

  def num_edges(self, edge_type):
    return self._edge_index_dict[edge_type].shape[1]

  @property
  def nodes_dict(self):
    return self._nodes_dict

  @property
  def edge_index_dict(self):
    return self._edge_index_dict

  @property
  def edges_dict(self):
    return self._edges_dict

  @property
  def keys(self):
    r"""Returns all names of graph attributes."""
    keys = [key for key in self.__dict__.keys() if self[key] is not None]
    keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
    return keys

  @property
  def node_types(self):
    """Returns all node types of the heterogeneous subgraph."""
    return list(self._nodes_dict.keys())

  @property
  def edge_types(self):
    """Returns all edge types of the heterogeneous subgraph."""
    return list(self._edge_index_dict.keys())

  def __getitem__(self, key):
    return getattr(self, key, None)

  def __setitem__(self, key, value):
    setattr(self, key, value)
```

### Tensor数据对象
对应nn/tf/data/


我们将一个GSL的结果首先组织成`Data` dict或者使用induce_func组成成`SubGraph`s, 这时候数据还是numpy格式。接着我们使用`tf.data.Dataset`的`from_generator`函数来完成numpy到tensor的转换。我们内置了一个`Dataset` 对象完成这样的转换。

#### Dataset
`Dataset`使用`from_generator`方法，完成numpy到tensor格式的转换，对外提供一个initializable的`iterator`，以及`get_data_dict`, `get_egograph`, `get_batchgraph`的接口。

在使用SubGraph/HeterSubGraph格式时需要提供一个inducer，完成GSL采样结果到SubGraph/HeterSubGraph的转换，具体参考examples/tf/sage, examples/tf/seal, examples/tf/bipartite_sage里的inducer例子。

```python
class Dataset(object):
  """`Dataset` object is used to convert GSL query results to Tensor format
  `Data`. It provides methods to get raw `Data` dict, `EgoGraph`, `SubGraph` 
  and `HeteroSubGraph`.

  Args:
    query: GSL query.
    window: dataset capacity.
    inducer: A `SubGraphInducer` instance to generate SubGraph/HeteroSubGraph.
  """
  def __init__(self, query, window=10, inducer=None, **kwargs):

  @property
  def iterator(self):
    return self._iterator
  
  def get_data_dict(self):
    """get a dict of tensor format `Data` corresponding the given query.
    Keys of the dict is the aliaes in query.
    """
    return self._rds.build_data_dict(list(self._values))

  def get_egograph(self, source, neighbors=None):
    """ Origanizes the data dict as EgoGraphs and then check and return
    the specified `EgoGraph`.
    Args:
      source(str): alias of centric vertices.
      neighbors(list of str): alias of neighbors at each hop.
        Default `None`: automatically generating the positive neighbors for
        centric vertices. It requires that each hop only has one postive
        downstream in GSL.
        Given list of string: the alias of each hop in GSL. The list must
        follow the order of traverse in GSL, and each one should be the postive
        or negative downstream for the front.
    """

  def get_batchgraph(self):
    """get `BatchGraph`/`HeteroBatchGraph`s. 
    """
	...
    return pos_graph, neg_graph
```


- `get_data_dict`: 将任意一个GSL的结果，转换成一个`Data` dict的形式。dict的key是GSL里的alias, value是对应的一个GSL节点的结果的tensor格式的`Data`。
- `get_egograph`：将包含fix-sized的采样的GSL的结果，转换成`EgoGraph`的格式。 可以通过alias来获取某个ego的`EgoGraph`子图
- `get_batchgraph`： 将包含full neighbor sampler的GSL的结果，转换成`BatchGraph`的格式，目前该方法还是experimental的，只支持从边遍历的一跳full neighbor采样，而且对GSL里的alias也有严格要求，必须是SubKeys里的一种。

下面我们具体介绍这三种格式的数据。
​

#### Data dict

任意的GSL都可以转换成`Data` dict的形式，dict的key是GSL里的alias, value是对应的一个GSL节点的结果的tensor格式的`Data`。
​

#### EgoGraph

`EgoGraph`表示一个由中心节点和其k-hop邻居构成的子图。提供了src(), hop_node(), hop_edge()等接口来获取中心节点，某跳的邻居点和某跳的邻居边。

```python
class EgoGraph(object):
  """ `EgoGraph` is a basic data structure used to describe a sampled graph. 
  It constists of src `Data` and src's neighbors(nodes and edges) `Data`.
  The `EgoGraph` is mainly used to represent subgraphs generated by fixed-size 
  neighbor sampling, in which the data can be efficiently organized in dense 
  format and the model can be computed using the dense operators.

  Args:
    src: A `Data`/Tensor object used to describe the centric nodes.
    nbr_nodes: A list of `Data`/Tensor instance to describe neighborhood nodes.
    node_schema: A list of tuple to describe the FeatureSpec of src and
      neighbor nodes. Each tuple is formatted with (name, spec), in which `name`
      is node's type, and `spec` is a FeatureSpec object. Be sure that
      `len(node_schema) == len(neighbors) + 1`.
    nbr_nums: A list of number of neighbor nodes per hop.
    nbr_edges: A list of `Data`/Tensor instance to describe neighborhood edges.
    edge_schema: A list of tuple to describe the `FeatureSpec` of neighbor edges.
  """
  def __init__(self,
               src,
               nbr_nodes,
               node_schema,
               nbr_nums,
               nbr_edges=[],
               edge_schema=[],
               **kwargs):
    self._src = src
    self._nbr_nodes = nbr_nodes
    self._node_schema = node_schema
    self._nbr_edges = nbr_edges
    self._edge_schema = edge_schema
    if self._node_schema is not None:
      assert len(self._node_schema) == len(nbr_nums) + 1
    self._nbr_nums = np.array(nbr_nums)

  @property
  def src(self):
    return self._src

  @property
  def node_schema(self):
    return self._node_schema

  @property
  def nbr_nodes(self):
    return self._nbr_nodes

  @property
  def nbr_nums(self):
    return self._nbr_nums

  @property
  def edge_schema(self):
    return self._edge_schema

  @property
  def nbr_edges(self):
    return self._nbr_edges

  def hop_node(self, i):
    """ Get the hop ith neighbors nodes of centric src, where i starts 
    from zero. The return value is a tensor with shape 
    [batch_size * k_1 *...* k_i, dim], where k_i is the expand neighbor 
    count at hop i and dim is the sum of all feature dimensions, which 
    may be different due to kinds of vertex types.
    """
    return self._nbr_nodes[i]
  
  def hop_edge(self, i):
    if len(self._nbr_edges) == 0:
      raise ValueError("No edge data.")
    return self._nbr_edges[i]
```
​

#### BatchGraph
为了高效批量训练，我们将一个batch的`SubGraph`s合并成`BatchGraph`的格式，`BatchGraph`继承自`SubGraph`, 提供`num_graphs`获取SubGraphs个数的接口，同时提供了`graph_node_offsets` 和 `graph_edge_offsets` 接口来表示合并后的每个`SubGraph`里的点和边的offset。


```python
class BatchGraph(SubGraph):
  """A BatchGraph object, which represents a batch of `SubGraph`s.
  Nodes, edges in subgraphs are concatenated together and their offsets 
  are recorded with `graph_node_offsets` and `graph_edge_offsets`. The
  `edge_index` of subgraph is remapped according to the order offset of 
  each subgrpah and then form as a new `edge_index`.

  Args:
    edge_index: concatenated edge_index of `SubGraph`s.
    nodes: A `Data`/Tensor object denoting concatenated nodes of `SubGraph`s 
      with shape [batch_size, attr_num].
    node_schema: A (name, Decoder) tuple used to describe 
      the nodes' feature or a list of such tuple to describe src and dst nodes'
      feature for heterogeneous graph.
    graph_node_offsets: indicates the nodes offset of each `SubGraph`.
    edges: A `Data`/Tensor object denoting concatenated edges of `SubGraph`s.
    node_schema: A (name, Decoder) tuple used to describe the edges' feature.
    graph_edge_offsets: indicates the edges offset of each `SuGraph`.
    additional_keys: A list of keys used to indicate the additional data. Note 
      that these keys must not contain the above args. 
      Note that we require this argument in order to keep the correct order of 
      the additional data when generating Tensor format of `BatchGraph`.
  """
  def __init__(self, edge_index, nodes, node_schema, graph_node_offsets,
               edges=None, edge_schema=None, graph_edge_offsets=None, 
               additional_keys=[], **kwargs)
   
  @property
  def num_nodes(self):
    if isinstance(self._nodes.ids, np.ndarray):
      return self._nodes.ids.size
    else:
      return self._nodes.ids.shape.as_list()[0]

  @property
  def num_edges(self):
    if isinstance(self._edge_index, np.ndarray):
      return self._edge_index.shape[1]
    else:
      return self._edge_index.shape.as_list()[1]

  @property
  def num_graphs(self):
    """number of SubGraphs.
    """
    if isinstance(self.graph_node_offsets, np.ndarray):
      return np.amax(self.graph_node_offsets) + 1
    else:
      return tf.reduce_max(self.node_graph_index) + 1

  @property
  def graph_node_offsets(self):
    return self._graph_node_offsets

  @property
  def graph_edge_offsets(self):
    return self._graph_edge_offsets

  @property
  def node_schema(self):
    return self._node_schema

  @property
  def edge_schema(self):
    return self._edge_schema

  @property
  def additional_keys(self):
    return self._additional_keys
```


#### HeteroBatchGraph(since v1.1.0)

```
class HeteroBatchGraph(HeteroSubGraph):
  """A HeteroBatchGraph object, which represents a batch of `HeteroSubGraph`s.
  Each type of Nodes, edges in subgraphs are concatenated together and their 
  offsets are recorded with `graph_node_offsets_dict` and 
  `graph_edge_offsets_dict`.

  Args:
    edge_index_dict: concatenated edge_index_dict of `HeteroSubGraph`s 
      according keys.
    nodes_dict: A dict of `Data`/Tensor object denoting concatenated nodes.
    node_schema_dict: A dict of {name: Decoder} to describe feature of each
      type of node.
    graph_node_offsets_dict: A dict, keys indicate the type of nodes, 
      values indicate the offset of each HeteroSubgraph nodes.
    edges_dict: A dict of `Data`/Tensor object denoting concatenated edges.
    edge_schema_dict: A dict of {name: Decoder} to describe feature of each
      type of edge. The edges_dict is not None when Decoder is not None.
    graph_edge_offsets_dict: A dict, keys indicate the type of edges, 
      values indicate the offset of each HeteroSubgraph edges.
  """
  def __init__(self, edge_index_dict, nodes_dict, node_schema_dict, 
               graph_node_offsets_dict, edges_dict=None, edge_schema_dict=None, 
               graph_edge_offsets_dict=None, **kwargs):
    super(HeteroBatchGraph, self).__init__(edge_index_dict, nodes_dict)
    self._edge_index_dict = edge_index_dict
    self._nodes_dict = nodes_dict
    self._node_schema_dict = node_schema_dict
    self._edges_dict = edges_dict
    self._edge_schema_dict = edge_schema_dict
    self._graph_node_offsets_dict = graph_node_offsets_dict
    self._graph_edge_offsets_dict = graph_edge_offsets_dict
    for key, item in kwargs.items():
      self[key] = item

  def num_nodes(self, node_type):
    nodes = self._nodes_dict[node_type]
    if isinstance(nodes.ids, np.ndarray):
      return nodes.ids.size
    else:
      return nodes.ids.shape.as_list()[0]

  def num_edges(self, edge_type):
    edge_index = self._edge_index_dict[edge_type]
    if isinstance(edge_index, np.ndarray):
      return edge_index.shape[1]
    else:
      return edge_index.shape.as_list()[1]

  @property
  def num_graphs(self):
    """number of SubGraphs.
    """
    graph_node_offsets = next(iter(self.graph_node_offsets_dict))
    if isinstance(graph_node_offsets, np.ndarray):
      return np.amax(graph_node_offsets) + 1
    else:
      return tf.reduce_max(graph_node_offsets) + 1

  @property
  def graph_node_offsets_dict(self):
    return self._graph_node_offsets_dict

  @property
  def graph_edge_offsets_dict(self):
    return self._graph_edge_offsets_dict

  @property
  def node_schema_dict(self):
    return self._node_schema_dict

  @property
  def edge_schema_dict(self):
    return self._edge_schema_dict
```

## 特征处理
对应nn/tf/data/
​

### 特征处理模块
上面我们完成了GSL的结果到tensor的转换，下面我们介绍特征处理模块。一般模型处理时需要输入的数据为一个连续的向量。在实际生产中，节点的特征往往包括int, float，string等连续、离散、多值特征，因此需要将这些特征处理成一个连续的向量。具体来说主要是对离散和多值特征通过`tf.nn.embedding_lookup`转换成一个连续的向量，然后和连续特征拼接在一起作为节点的输入特征。
​

#### FeatureColumn
`FeatureColumn`提供了对不同类型特征的处理。包括连续特征`NumericColumn`, 离散特征`EmbeddingColumn`, 多值离散特征`SparseEmbeddingColumn`。为了加速embedding_lookup过程，我们同时封装了`FusedEmbeddingColumn`，将相同dimension的数据合并在一起进行embedding_lookup。
对于embedding variable比较大的情况，我们使用了`tf.min_max_variable_partitioner`（默认）进行varibale的切分。

```python
class FeatureColumn(Module):
  """ Transforms raw features to dense tensors. For continuous features, just 
  return the original values, for categorical features, embeds them to dense
  vectors.

  For example, each 'user' vertex in the graph contains 6 attributes splited by
  ':', which looks like '28:0:0.2:Hangzhou:1,5,12:1000008'. To handle such a
  vertex, 6 `FeatureColumn` objects are needed, each of which will return a
  dense value. And then we will concat all the dense values together to get the
  representation of this vertex.

  Each feature can be configured differently. The first two features, 28 and 0,
  are categorical, and both of them will be encoded into continuous spaces with
  dimension 12. To improve the efficiency, we can fuse the two spaces together
  to minimize the communication frequence when encoding. If the shapes of raw
  spaces are [100, 12] and [50, 12], we will get one space with shape [150, 12]
  after fusion.

  The third feature is 0.2, we just return it as a numeric feature.

  The fourth feature is a string, which need to be transformed into an integer
  and then encoded with a continuous space.

  The fifth feature is a multi-value splited by ','. The count of elements is
  not fixed. We need to encode each value into a continuous space and merge
  them together.

  The last feature is a big integer, and just transform it into a continuous
  space.

  All of the above features will be handled by different FeatureColumns, and
  then concatenated by a FeatureGroup.
  """

  def __init__(self):
    pass

  def forward(self, x):
    raise NotImplementedError


class PartitionableColumn(FeatureColumn):
  """ `PartitionableColumn` uses `tf.min_max_variable_partitioner` to 
  partition the embedding varibles. Note that the `conf.emb_max_partitions` 
  must be provided when using partitioner.
  """
  def _partitioner(self):
    max_parts = conf.emb_max_partitions
    if max_parts is not None:
      return tf.min_max_variable_partitioner(
          max_partitions=max_parts, min_slice_size=conf.emb_min_slice_size)
    else:
      return None


class NumericColumn(FeatureColumn):
  """ Represents real valued or numerical features.
  Args:
    name: A unique string identifying the input feature.
    normalizer_func: If not `None`, a function that can be used to normalize 
      the value of the tensor. Normalizer function takes the input `Tensor` 
      as its  argument, and returns the output `Tensor`. 
      (e.g. lambda x: (x - 1.0) / 2.0). 
  """
  def __init__(self, name, normalizer_func=None)


class EmbeddingColumn(PartitionableColumn):
  """ Uses embedding_lookup to embed the categorical features.
  Args:
    name: A unique string identifying the input feature.
    bucket_size: The size of the embedding variable.
    dimension: The dimension of the embedding.
    need_hash: Whether need hash the input feature.
  """
  def __init__(self, name, bucket_size, dimension, need_hash=False)
    

class DynamicEmbeddingColumn(PartitionableColumn):
  """ EmbeddingColumn with dynamic bucket_size.
  """
  def __init__(self, name, dimension, is_string=False):


class FusedEmbeddingColumn(PartitionableColumn):
  """ Fuses the input feature with the same dimension setting and then
  lookups embeddings.
  Args:
    name: A unique string identifying the input feature.
    bucket_list: A list of the size of the embedding variable.
    dimension: The dimension of the embedding.
  """
  def __init__(self, name, bucket_list, dimension)


class SparseEmbeddingColumn(PartitionableColumn):
  """ Uses sparse_embedding_lookup to embed the multivalent categorical 
  feature which is split with delimiter.
  Args:
    name: A unique string identifying the input feature.
    bucket_size: The size of the embedding variable.
    dimension: The dimension of the embedding.
    delimiter: The delimiter of multivalent feature.
  """
  def __init__(self, name, bucket_size, dimension, delimiter)


class DynamicSparseEmbeddingColumn(PartitionableColumn):
  """ SparseEmbeddingColumn with dynamic bucket_size.
  """
  def __init__(self, name, dimension, delimiter)
```


#### FeatureHandler
一个FeatureColumn处理一列特征，多个 `FeatureColumn` 聚集在一起形成 `FeatureGroup` ，顶点和边的特征根据排列顺序可能划分到多个 `FeatureGroup` ，划分逻辑由 `FeatureHandler` 实现。`FeatureHandler` 接收一个完整的顶点或边的`feature_spec` （见“图操作接口->数据源->Decoder定义”），返回对应顶点或边的拼接后的向量化特征。**向量的维度需要在构图时通过decoder指定，**`**FeatureHandler**`**返回的结果的dimension和decoder里指定的特征配置对应。**
​
```python
class FeatureHandler(Module):
  """Encodes the input features of `Data` using `FeatureSpec`.
  For efficiency, we group the features into `FeatureGroup` accroding to 
  the `FeatureSpec` and then encode each `FeatureGroup` and merge their 
  outputs as the final output.

  Args:
    name: A unique string.
    feature_spec: A `FeatureSpec` object to describe the input feature 
      of `Data`.
    fuse_embedding: Whether fuses the input features of the same 
      specified dimension before feature encoding(embedding lookup).
  """
  def __init__(self, name, feature_spec,
               fuse_embedding=True)
```
​

### transform
对`EgoGraph`和`BatchGraph`，我们提供了`transform`函数完成上述特征的处理和变化。`transform`里会构建一个`FeatureHandler`，进行特征的处理，同时，`transform`也支持传一个`transform_func`进行一些特征预处理操作。


#### EgoGraph的transform

```python
  def transform(self, transform_func=None):
    """transforms `EgoGraph`. Default transformation is encoding nodes feature 
    to embedding.
    Args:
      transform_func: A function that takes in an `EgoGraph` object and returns 
        a transformed version. 
    """
    if self.node_schema is None:
      return self

    assert len(self.node_schema) == (len(self.nbr_nodes) + 1)

    s = self.node_schema[0]
    vertex_handler = FeatureHandler(s[0], s[1])
    vertex_tensor = vertex_handler.forward(self.src)

    neighbors = []
    for i, nbr in enumerate(self.nbr_nodes):
      s = self.node_schema[i + 1]
      neighbor_handler = FeatureHandler(s[0], s[1])
      neighbor_tensor = neighbor_handler.forward(self.nbr_nodes[i])
      neighbors.append(neighbor_tensor)

    return EgoGraph(vertex_tensor, neighbors, None, self.nbr_nums)
```
​

#### BatchGraph的transform

```python
  def transform(self, transform_func=None):
    """transforms `BatchGraph`. Default transformation is encoding 
    nodes feature to embedding.
    Args:
      transform_func: A function that takes in an `BatchGraph` object 
        and returns a transformed version. 
    """
    if self.node_schema is None:
      return self
    vertex_handler = FeatureHandler(self.node_schema[0],
                                    self.node_schema[1].feature_spec)
    node = Data(self.nodes.ids, 
                self.nodes.int_attrs, 
                self.nodes.float_attrs, 
                self.nodes.string_attrs)
    node_tensor = vertex_handler.forward(node)
    graph = BatchGraph(self.edge_index, node_tensor, 
                       self.node_schema, self.graph_node_offsets,
                       additional_keys=self.additional_keys)
    for key in self.additional_keys:
      graph[key] = self[key]
    return graph
```
​

在构图时配置了`Decoder`的`attr_types`和`attr_dims`的情况下，我们在用`Dataset`生成`EgoGraph`或`BatchGraph`时会自动取得`Decoder`对应的`feature_spec`，在模型前向过程中，可以直接调用`transform`得到特征转换后的`EgoGraph`或`BatchGraph`。








