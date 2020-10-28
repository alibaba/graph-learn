# Graph Sampling

<a name="fOsGk"></a>

# 1 Introduction
Graph sampling is an effective way to deal with very large scale graphs, which has been widely adopted by mainstream Graph Neural Network models such as [GraphSage](https://arxiv.org/abs/1706.02216). In pratice, sampling not only reduces the data scale but also achieves data alignment, which is conducive to the efficient processing of the Tensor-Based computing framework. <br />

<br />We summarized **2 types** sampling operations according to user needs: **neighbor sampling(NEIGHBORHOOD)**, **negative sampling(NEGATIVE)**. Neighbor sampling is retrieving many-hop neighbors of the given input vertices which could be produced by external data sources or another graph traversal's output. Those sampled neighbors could be used to construct the receptive field in [GCN](https://arxiv.org/abs/1609.02907) theory. Negative sampling is retrieving vertices which are not directly connect to the given input vertices. In pratice, negative sampling is an important method for supervised learning.<br />

<br />Each type of sampling operation has different sampling strategies, such as random, weighted, etc. Benefiting from the rich application scenarios, we have implemented more than 10 kinds of ready-made sampling operators, and also export the operator's programming interface, allowing users to customize their own sampling operators to meet the needs of the rapidly developing GNN. This chapter introduces neighbor sampling, and negative sampling will be described in detail in the next chapter. In addition, the sub-graph sampling methods proposed in the recent AI top conferences is under development.<br />

<a name="gvEnk"></a>
# 2 Usage
<a name="OI8t6"></a>
## 2.1 Interface
The sampling operator takes meta-path and sampling number as input which supports for sampling any heterogeneous graph and arbitrary number of hops. The sampling results are organized into `Layers` objects, and n-th `Layer` is the n-th hop sampling results. `Nodes` and `Edges` objects are member variables in each `Layer` object. A sampling operation can be divided into the following 3 steps:

- Define the sampling oprator by invoking `g.neighbor_sampler()` to get the `NeighborSampler` object `S`
- Invoke `S.get(ids)` to get the neighbor `Layers` object `L` of the vertex;
- Invoke `L.layer_nodes(i)`, `L.layer_edges(i)` to get the vertices and edges of the i-th hop neighbors;


```python
def neighbor_sampler(meta_path, expand_factor, strategy="random"):
"""
Args:
  meta_path(list):     List of edge_type(string). Stand for paths of neighbor sampling.
  expand_factor(list): List of int. The i-th element represents the number of neighbors sampled by the i-th hop; the length must be consistent with meta_path
  strategy(string):    Sampling strategy. Please refer to the detailed explanation below.
Return:
  NeighborSampler object
"""
```
```python
def NeighborSampler.get(ids):
""" Sampling multi-hop neighbors of given ids.
Args:
  ids(numpy.ndarray): 1-d int64 array
Return:
  Layers object
"""
```
The returned result of the sampling is a `Layers` object, which means "the neighbor vertices of the source vertex, and the edges between the source vertex and the neighbor vertices". The shape of each `Layer`'s ids is two-dimensional. More specifically, ids shape=**[expanded size of ids of the previous layer, number of samples in the current layer]**.
```python
def Layers.layer_nodes(layer_id):
""" Get the `Nodes` of the i-th layer, layer_id starts from 1. """
    
def Layers.layer_edges(layer_id):
""" Get the `Edges` of the i-th layer, layer_id starts from 1. """
```

<br />In GSL, refer to the `g.out*` and `g.in*` families of interfaces. E.g.
```python
# Sampling 1-hop neighbor vertices
g.V().outV().sample(count).by(strategy)

# Sampling 2-hop neighbor vertices
g.V().outV().sample(count).by(strategy).outV().sample(count).by(strategy)
```

<a name="j0egY"></a>
## 2.2 Example
Consider a query that samples the 2-hop neighbors of a given vertex of user type as shown in the picture below. The returned result is layers which contains two layers: layer1 and layer2. **The index of layer starts from 1**, which means layer1 corresponds to 1-hop neighbor and so on.

<div align=center><img src ="images/2-hop-sampling.png" /> </div>

```python
s = g.neighbor_sampler(["buy", "i2i"], expand_factor=[2, 2])
l = s.get(ids) # input ids: shape=(batch_size)

# Nodes object
# shape=(batch_size, expand_factor[0])
l.layer_nodes(1).ids
l.layer_nodes(1).int_attrs

 # Edges object
 # shape=(batch_size *  expand_factor[0],  expand_factor[1])
l.layer_edges(2).weights
l.layer_edges(2).float_attrs
```

<a name="UpHHt"></a>
# 3 Sampling Strategy
GL currently supports the following sampling strategies, which are possible values of the `strategy` parameter when creating the `NeighborSampler` object.

| **strategy** | **Description** |
| --- | --- |
| edge_weight | Sampling with edge weight as probability |
| random | Random sampling with replacement |
| random_without_replacement | Random sampling without replacement. Refer to the padding rule when the number of neighbors is not enough |
| topk | Return the top k neighbors with edge weight. Refer to the padding rule when the number of neighbors is not enough |
| in_degree | Sampling with vertex in-degree probability |
| full | Return all neighbors. The expand_factor parameter has no effect on this strategy. The result object is SparseNodes or SparseEdges, see [数据查询](graph_query_cn.md#FPU74) for detailed decription of those objects. |

<br />Padding rules: the returned value need to be padded in some way when the amount of data required for sampling is not enough. By default, `default_neighbor_id` is used to pad missing `id` and its default value is 0, users can set it by calling `gl.set_default_neighbor_id(xx)`. If cyclically padding is needed, that is, padding with existing neighbor ids instead of `default_neighbor_id`, users need to set the padding mode to `gl.CIRCULAR` by calling `gl.set_padding_mode(gl.CIRCULAR)`.
