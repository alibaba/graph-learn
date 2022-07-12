## 模型层

对应nn/tf/layers和nn/tf/model
​

大部分GNNs算法遵从递归的消息传递/邻居聚合的范式，因此可以类似一般的DNN，抽象出层的概念，来表示一次消息传递操作。目前常见的GNNs都是图卷积神经网络，因此我们抽象了若干conv层表示一次图卷积过程。对于EgoGraph, 为了方便地对异构图进行消息传递，我们在conv层之上抽象了layer的概念，来表示一个子图的一次完整消息传递过程。基于这些conv或者layer，可以很方便得构建出一个GNNs模型，我们内置了若干常见的模型，也欢迎大家贡献，补充更多的GNNs模型。


### Layers
对SubGraph/BatchGraph，我们提供了若干`SubConv`层，对EgoGraph提供了`EgoConv`和`EgoLayer`。
​

#### SubGraph based layer


SubGraph的一次卷积可以通过edge_index和node_vec来计算，我们将基本的卷积层定义为`SubConv`

- SubConv

```python
class SubConv(Module):
  __metaclass__ = ABCMeta

  @abstractmethod
  def forward(self, edge_index, node_vec, **kwargs):
    """
    Args:
      edge_index: A Tensor. Edge index of subgraph.
      node_vec: A Tensor. node feature embeddings with shape
      [batchsize, dim].
    Returns:
      A tensor. output embedding with shape [batch_size, output_dim].
    """
    pass
```


基于SubConv基类，可以实现不同的图卷积层。
​

- GCNConv

```python
class GCNConv(SubConv):
  def __init__(self, in_dim, out_dim,
               normalize=True,
               use_bias=False,
               name='')
```
 
- GATConv

```python
class GATConv(SubConv):
  """multi-head GAT convolutional layer.
  """
  def __init__(self,
               out_dim,
               num_heads=1,
               concat=False,
               dropout=0.0,
               use_bias=False,
               name='')
```
 
- SAGEConv

```python
class SAGEConv(SubConv):
  def __init__(self, in_dim, out_dim,
               agg_type='mean', # sum, gcn, mean
               normalize=False,
               use_bias=False,
               name='')
```

- HeteroConv

```python
class HeteroConv(Module):
  """ Handles heterogeneous subgraph(`HeteroSubGraph`) convolution.
  
  This layer will perform the convolution operation according to the 
  specified edge type and its corresponding convolutional layer(`conv_dict`). 
  If multiple edges point to the same destination node, their results 
  will be aggregated according to `agg_type`.

  Args:
    conv_dict: A dict containing `SubConv` layer for each edge type.
    agg_type: The aggregation type used to specify the result aggregation 
      method when the same destination node has multiple edges.
      The optional values are: `sum`, `mean`, `min`, `max`, the default 
      value is `mean`. 
  """
  def __init__(self, conv_dict, agg_type='mean'):
    super(HeteroConv, self).__init__()
    self.conv_dict = conv_dict
    self.agg_type = agg_type

  def forward(self, edge_index_dict, node_vec_dict, **kwargs):
    """
    Args:
      edge_index_dict: A dict containing edge type to edge_index mappings.
      node_vec_dict: A dict containing node type to node_vec mappings.
    Returns:
      A dict containing node type to output embedding mappings.
    """
    out_dict = defaultdict(list)
    for edge_type, edge_index in edge_index_dict.items():
      h, r, t = edge_type
      if edge_type not in self.conv_dict:
        continue
      if h == t:
        out = self.conv_dict[edge_type](edge_index, node_vec_dict[h])
      else:
        out = self.conv_dict[edge_type](edge_index, [node_vec_dict[h], node_vec_dict[t]])
      out_dict[t].append(out)
    
    for k, v in out_dict.items():
      if len(v) == 1:
        out_dict[k] = v[0]
      else:
        out_dict[k] = getattr(tf.math, 'reduce_' + self.agg_type)(v, 0)
    
    return out_dict
```


#### EgoGraph based layers
对于EgoGraph， 我们将一次k+1 hop的邻居到k hop的邻居的聚合过程定义为一个`EgoConv`


- EgoConv

```python
class EgoConv(Module):
  """Represents the single propagation of 1-hop neighbor to centeric nodes."""
  __metaclass__ = ABCMeta

  @abstractmethod
  def forward(self, x, neighbor, expand):
    """ Update centeric node embeddings by aggregating neighbors.
    Args:
      x: A float tensor with shape = [batch_size, input_dim].
      neighbor: A float tensor with shape = [batch_size * expand, input_dim].
      expand: An integer, the neighbor count.

    Return:
      A float tensor with shape=[batch_size, output_dim].
    """
```
 
基于`EgoConv`可以实现各自图卷积层
​

- EgoSAGEConv

```python
class EgoSAGEConv(EgoConv):
  """ GraphSAGE. https://arxiv.org/abs/1706.02216.

  Args:
    name: A string, layer name.
    in_dim: An integer or a two elements tuple. Dimension of input features.
      If an integer, nodes and neighbors share the same dimension.
      If an tuple, the two elements represent the dimensions of node features
      and neighbor features.
      Usually, different dimensions happen in the heterogeneous graph. Note that
      for 'gcn' agg_type, in_dim must be an interger cause gcn is only for 
      homogeneous graph.
    out_dim: An integer, dimension of the output embeddings. Both the node
      features and neighbor features will be encoded into the same dimension,
      and then do some combination.
    agg_type: A string, how to merge neighbor values. The optional values are
      'mean', 'sum', 'max' and 'gcn'.
    use_bias: A boolean, whether add bias after computation.
  """

  def __init__(self,
               name,
               in_dim,
               out_dim,
               agg_type="mean",
               use_bias=False,
               **kwargs)
```

- EgoGATConv

```python
class EgoGATConv(EgoConv):
  """ Graph Attention Network. https://arxiv.org/pdf/1710.10903.pdf.

  Args:
    name: A string, layer name.
    in_dim: An integer or a two elements tuple. Dimension of input features.
      If an integer, nodes and neighbors share the same dimension.
      If an tuple, the two elements represent the dimensions of node features
      and neighbor features.
      Usually, different dimensions happen in the heterogeneous graph.
    out_dim: An integer, dimension of the output embeddings. Both the node
      features and neighbor features will be encoded into the same dimension,
      and then do some combination.
    use_bias: A boolean, whether add bias after computation.
  """

  def __init__(self,
               name,
               in_dim,
               out_dim,
               num_head=1,
               use_bias=False,
               attn_dropout=0.0,
               **kwargs)
```


- EgoGINConv

```python
class EgoGINConv(EgoConv):
  """ GIN. https://arxiv.org/abs/1810.00826.

  Args:
    name: A string, layer name.
    in_dim: An integer or a two elements tuple. Dimension of input features.
      If an integer, nodes and neighbors share the same dimension.
      If an tuple, the two elements represent the dimensions of node features
      and neighbor features.
      Usually, different dimensions happen in the heterogeneous graph.
    out_dim: An integer, dimension of the output embeddings. Both the node
      features and neighbor features will be encoded into the same dimension,
      and then do some combination.
    use_bias: A boolean, whether add bias after computation.
  """

  def __init__(self,
               name,
               in_dim,
               out_dim,
               eps=0.0,
               use_bias=False,
               **kwargs)
```


#### EgoLayer
上述的`EgoConv`层只是表示了k+1跳到k跳邻居的聚合过程，对于一个由ego和`K`跳邻居组成的`EgoGraph`来说，一次全图的消息传递需要对EgoGraph里所有的相邻的邻居对进行`EgoConv`的前向操作，即对k ={0, 1, 2, ... `K-1`}, 都进行k+1到k跳的聚合。我们将这样由若干1跳邻居聚合`EgoConv`构成的一次EgoGraph全图的消息传递过程用`EgoLayer`表示。
`EgoLayer`和`EgoConv`的关系如下图所示。`EgoLayer`表示的是一次EgoGraph子图上的消息传递，而`EgoConv`表示的是相邻的k跳和k+1跳邻居的一次消息传递。对于2跳邻居构成的`EgoGraph`，有2个`EgoLayer`，第一个`EgoLayer`包含2个`EgoConv`，第二个`EgoLayer`包含1个`EgoConv`。可以看出由`EgoConv`组成的`EgoLayer`自然支持异构图的meta-path消息传递过程，对于同构图，只需要复用`EgoConv`即可。
​

![egolayer](../../../images/egolayer.png)


```python
class EgoLayer(Module):
  """Denotes one convolution of all nodes on the `EgoGraph`. 
  For heterogeneous graphs, there are different types of nodes and edges, so 
  one convolution process of graph may contain serveral different aggregations
  of 1-hop neighbors based on node type and edge type. We denote `EgoConv` as 
  a single propogation of 1-hop neighbor to centric nodes, and use `EgoLayer` to
  represent the entire 1-hop propogation of `EgoGraph`.
  """
  def __init__(self, convs):
    super(EgoLayer, self).__init__()
    self.convs = convs

  def forward(self, x_list, expands):
    """ Update node embeddings.

    x_list = [nodes, hop1, hop2, ... , hopK-1, hopK]
               |   /  |   /  |   /        |    /
               |  /   |  /   |  /         |   /
               | /    | /    | /          |  /
    output = [ret0,  ret1, ret2, ... , retK-1]

    Args:
      x_list: A list of tensors, representing input nodes and their K-hop neighbors.
        If len(x_list) is K+1, that means x_list[0], x_list[1], ... , x_list[K]
        are the hidden embedding values at each hop. Tensors in x_list[i] are
        the neighbors of that in x_list[i-1]. In this layer, we will do
        convolution for each adjencent pair and return a list with length K.

        The shape of x_list[0] is `[n, input_dim_0]`, and the shape of x_list[i]
        is `[n * k_1 * ... * k_i, input_dim_i]`, where `k_i` means the neighbor
        count of each node at (i-1)th hop. Each `input_dim_i` must match with
        `input_dim` parameter when layer construction.

      expands: An integer list of neighbor count at each hop. For the above
        x_list, expands = [k_1, k_2, ... , k_K]

    Return:
      A list with K tensors, and the ith shape is
      `[n * k_1 * ... * k_i, output_dim]`.
    """
```


### Model


#### SubGraph based model
基于`SubConv`层，可以很方便地构建出一个GNNs模型，我们内置了一些常见的GNNs模型。所有模型都需要实现`forward`过程， `forward`接受`BatchGraph`对象，返回最后的embedding。目前只支持同构图从边遍历，并返回src和dst的embedding。

```python
def forward(self, batchgraph)
```

- GCN

```python
class GCN(Module):
  def __init__(self,
               batch_size,
               input_dim,
               hidden_dim,
               output_dim,
               depth=2,
               drop_rate=0.0,
               **kwargs)
```
​

- GAT

```python
class GAT(Module):
  def __init__(self,
               batch_size,
               hidden_dim,
               output_dim,
               depth=2,
               drop_rate=0.0,
               attn_heads=1,
               attn_drop=0.0,
               **kwargs)
```
​

- SAGE

```python
class GraphSAGE(Module):
  def __init__(self,
               batch_size,
               input_dim,
               hidden_dim,
               output_dim,
               depth=2,
               drop_rate=0.0,
               agg_type='mean', # sum, gcn, mean
               **kwargs)
```
​

- SEAL

```python
class SEAL(Module):
  def __init__(self,
               batch_size,
               input_dim,
               hidden_dim,
               output_dim,
               depth=2,
               drop_rate=0.0,
               agg_type='mean', # sum, gcn, mean
               **kwargs)
```


#### EgoGraph based model


基于`EgoConv`组成的`EgoLayer`，我们可以快速构建出GNN模型。由于`EgoLayer`支持一般的异构图，因此`EgoGraph` based GNN可以统一用如下的模型实现。
​

- EgoGNN

```python
class EgoGNN(Module):
  """ Represents `EgoGraph` based GNN models.

  Args:
    layers: A list, each element is an `EgoLayer`.
    bn_func: Batch normalization function for hidden layers' output. Default is
      None, which means batch normalization will not be performed.
    act_func: Activation function for hidden layers' output. 
      Default is tf.nn.relu.
    dropout: Dropout rate for hidden layers' output. Default is 0.0, which
      means dropout will not be performed. The optional value is a float.
  """

  def __init__(self,
               layers,
               bn_func=None,
               act_func=tf.nn.relu,
               dropout=0.0,
               **kwargs):
    super(EgoGNN, self).__init__()

    self.layers = layers
    self.bn_func = bn_func
    self.active_func = act_func
    self.dropout = dropout

  def forward(self, graph):
    """ Update node embeddings through the given ego layers.

    h^{i} is a list, 0 <= i <= n, where n is len(layers).
    h^{i} = [ h_{0}^{i}, h_{1}^{i}, h_{2}^{i}, ... , h_{n - i}^{i} ]

    For 3 layers, we need nodes and 3-hop neighbors in the graph object.
      h^{0} = [ h_{0}^{0}, h_{1}^{0}, h_{2}^{0}, h_{3}^{0} ]
      h^{1} = [ h_{0}^{1}, h_{1}^{1}, h_{2}^{1} ]
      h^{2} = [ h_{0}^{2}, h_{1}^{2} ]
      h^{3} = [ h_{0}^{3} ]

    For initialization,
      h_{0}^{0} = graph.src
      h_{1}^{0} = graph.hop_node{1}
      h_{2}^{0} = graph.hop_node{2}
      h_{3}^{0} = graph.hop_node{3}

    Then we apply h^{i} = layer_{i}(h^{i-1}), and h_{0}^{3} is the final returned value.

    Args:
      graph: an `EgoGraph` object.

    Return:
      A tensor with shape [batch_size, output_dim], where `output_dim` is the
      same with layers[-1].
    """
    graph = graph.transform() # feature transformation of `EgoGrpah`

    # h^{0}
    h = [graph.src]
    for i in range(len(self.layers)):
      h.append(graph.hop_node(i))

    hops = graph.nbr_nums
    for i in range(len(self.layers) - 1):
      # h^{i}
      current_hops = hops if i == 0 else hops[:-i]
      h = self.layers[i].forward(h, current_hops)
      H = []
      for x in h:
        if self.bn_func is not None:
          x = self.bn_func(x)
        if self.active_func is not None:
          x = self.active_func(x)
        if self.dropout and conf.training:
          x = tf.nn.dropout(x, keep_prob=1-self.dropout)
        H.append(x)
      h = H

    # The last layer
    h = self.layers[-1].forward(h, [hops[0]])
    assert len(h) == 1
    return h[0]
```
​

​

#### 其他
除了常见GNNs模型，对于一些常用的模块我们也封装了对应的模型，比如链接预测模块。
​

- LinkPredictor
链接预测模块里封装了若干dense层，对输入向量经过这些dense层后输出最终结果。

```python
class LinkPredictor(Module):
  """ link predictor.

  Args:
    name: The name of link predictor.
    input_dim: The Input dimension.
    num_layers: Number of hidden layers.
    active_fn: Activation function for hidden layers' output.
    dropout: Dropout rate for hidden layers' output. Default is 0.
  """

  def __init__(self,
               name,
               input_dim,
               num_layers,
               dropout=0.0)
    
  def forward(self, x):
    """
    Args:
      x: input Tensor of shape [batch_size, input_dim]
    Returns:
      logits: Output logits tensor with shape [batch_size]
    """
```


