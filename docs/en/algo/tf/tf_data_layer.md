## Data Layer
One of the main goals of the data layer is to provide a conversion of the numpy data stream returned by GraphLearn graph operations to the tensor data stream in TensorFlow1.x. We use the `from_generator` interface of `tf.data.Dataset` to accomplish such a conversion process. In addition, to facilitate the unified processing of node and edge features, we provide a feature processing interface to complete the processing of the original continuous features, discrete features, and multi-valued discrete features into a continuous vector as the input to the subsequent model. We describe the whole data layer construction process in the following order of data processing.
**Basic data object**, which describes the basic data objects of the model layer. **Tensor data object**, describing the data format of the TensorFlow tensor format.
**Feature Processing**, describing the interface for processing different types of raw features.


### Basic Data object
nn/data.py, nn/subgraph.py, nn/dataset.py


The return result of each graph operation of GraphLearn is a `Nodes` or `Edges` object in numpy ndarray format. To facilitate the processing of the model layer, we use a `Data` object to represent `Nodes` and `Edges` in a uniform way. Such a GSL query result can be represented by a `Data` dict, key is alias, value is the specific `Data` object.


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


#### SubGraph
To simplify the modeling of GNNs algorithm, we use `SubGraph` to represent a sampled subgraph, which consists of `edge_index`, `nodes` and `edges`. Since GSL does not currently provide the ability to directly sample the graph, an `induce_func` needs to be displayed when generating the SubGraph.

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




### Tensor Data Object
nn/tf/data/


We first organize the results of a GSL into `Data` dict or into `SubGraph`s using reduce_func, when the data is still in numpy format. Then we use the `from_generator` function of `tf.data.Dataset` to do the numpy to tensor conversion. We have a built-in `Dataset` object to do this conversion.

#### Dataset
`Dataset` uses the `from_generator` method to complete the conversion from numpy to tensor format, and provides an initializable `iterator` and interfaces to `get_data_dict`, `get_egograph`, `get_batchgraph`.

```python
class Dataset(object):
  """`Dataset` object is used to convert GSL query results to Tensor format
  `Data`. It provides methods to get raw `Data` dict and `EgoGraph`s 
  composed of `Data`, and also method to get `BatchGraph`s composed of 
  `SubGraph`s when induce_func is provided(not None).

  Args:
    query: GSL query, which must contain `SubKeys` as aliases.
    induce_func: `SubGraph` inducing function, it should be either 
      induce with edge or induce with node. The induce with edge function
      requires 4 args (src, dst, src_nbrs, dst_nbrs), and the induce with 
      node function requires 2 args (src, src_nbrs). 
      This function should be overridden when you need implement 
      your own SubGraph inducing procedure.
    induce_additional_spec: A dict to describe the additional data of 
      BatchGraph which is generated by the induce_func. Each key is the name 
      of additional data, and values is a list [types, shapes], which are
      tf.dtype and tf.TensorShape instance to describe the tensor format 
      types and shapes of additional data.
    window: dataset capacity.
  """
  def __init__(self, query, window=5,                
               induce_func=None,
               induce_additional_spec=None, 
               **kwargs)
 

  @property
  def iterator(self):
    return self._iterator

  def get_data_dict(self):
    """get a dict of tensor format `Data` corresponding the given query.
    Keys of the dict is the aliaes in query.
    """
    
    
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
    
  def get_batchgraph(self, alias):
    """get `BatchGraph`s by given alias. Alias must be an element 
    in `SubKeys`.
    """
```

- `get_data_dict`: convert any GSL result into a `Data` dict. key of the dict is the alias of the GSL, value is the tensor format `Data` of the result of a GSL node.
- `get_egograph`: converts the result of GSL containing fix-sized samples into `EgoGraph` format. You can get the `EgoGraph` of a certain ego by alias.
- `get_batchgraph`: convert the result of GSL containing full neighbor sampler into `BatchGraph` format.

We describe these three formats of data in detail below.​

#### Data dict

Any GSL can be converted into a `Data` dict, where the key of the dict is the alias in the GSL, and the value is the `Data` in tensor format corresponding to the result of a GSL node.
​

#### EgoGraph

`EgoGraph` represents a subgraph consisting of a central node and its k-hop neighbors. Interfaces such as src(), hop_node(), hop_edge() are provided to get the central node, the neighbor nodes of a certain hop and the neighbor edges of a certain hop.

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
    nbr_nums: A list of number of neighbor per hop.
    nbr_edges: A list of `Data`/Tensor instance to describe neighborhood edges.
    edge_schema: A list of tuple to describe the `FeatureSpec` of neighbor edges.
  """
  def __init__(self,
               src,
               nbr_nodes,
               node_schema,
               nbr_nums,
               nbr_edges=None,
               edge_schema=None,
               **kwargs)
   
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
    return self._nbr_nodes

  def hop_node(self, i):
    """ Get the hop ith neighbors nodes of centric src, where i starts 
    from zero. The return value is a tensor with shape 
    [batch_size * k_1 *...* k_i, dim], where k_i is the expand neighbor 
    count at hop i and dim is the sum of all feature dimensions, which 
    may be different due to kinds of vertex types.
    """
    return self._nbr_nodes[i]
  
  def hop_edge(self, i):
    if self._nbr_edges is not None:
      return self._nbr_edges[i]
    else:
      raise ValueError("No edge data.")
```
​

#### BatchGraph
For efficient batch training, we merge the `SubGraph`s of a batch into the format of `BatchGraph`, which inherits from `SubGraph`, providing the interface `num_graphs` to get the number of SubGraphs, and the interfaces `graph_node_ offsets` and `graph_edge_offsets` interfaces to represent the offset of the nodes and edges in each `SubGraph` after merging.


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


​

## Feature Processing
nn/tf/data/


### Feature processing module
Above we have completed the conversion of GSL results to tensor, below we introduce the feature processing module. General model processing requires the input data to be a continuous vector. In actual production, the features of nodes often include continuous, discrete and multi-valued features such as int, float, string, etc. Therefore, these features need to be processed into a continuous vector. Specifically, the discrete and multi-valued features are mainly converted into a continuous vector by `tf.nn.embedding_lookup`, and then stitched together with the continuous features as the node's input features.
​

#### FeatureColumn
`FeatureColumn` provides processing of different types of features. These include continuous features `NumericColumn`, discrete features `EmbeddingColumn`, and multi-valued discrete features `SparseEmbeddingColumn`. In order to speed up the embedding_lookup process, we also package `FusedEmbeddingColumn` to combine data of the same dimension together for embedding_lookup.
For the case of large embedding variable, we use `tf.min_max_variable_partitioner` (default) for varibale slicing.

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
A FeatureColumn deals with one column of features, and multiple `FeatureColumns` are clustered together to form a `FeatureGroup`, and the features of vertices and edges may be divided into multiple `FeatureGroup`s according to the order of arrangement, and the division logic is implemented by the `FeatureHandler`. The `FeatureHandler` receives the `feature_spec` of a complete vertex or edge (see "Data Source->Decoder Definition") and returns the stitched vectorized features of the corresponding vertex or edge. The dimension of the **vector needs to be specified by the decoder when constructing the graph.
The dimension of the result returned by `**FeatureHandler**` corresponds to the feature configuration specified in the decoder.
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
For `EgoGraph` and `BatchGraph`, we provide `transform` function to complete the processing and change of the above features. A `FeatureHandler` is constructed in `transform` for feature processing, and `transform` also supports passing a `transform_func` for some feature pre-processing operations.


#### transform of EgoGraph

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

#### transform of BatchGraph

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

If the `attr_types` and `attr_dims` of `Decoder` are configured when building the graph, we will automatically get the `feature_spec` corresponding to `Decoder` when generating `EgoGraph` or `BatchGraph` with `Dataset`, and we can directly call `transform` to get the feature-transformed `EgoGraph` or `BatchGraph` in the model forwarding process.








