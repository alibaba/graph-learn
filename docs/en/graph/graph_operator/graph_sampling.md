## Graph Sampling

### Introduction
Graph sampling is an effective means of processing very large graphs and has been widely used in programming paradigms represented by the [GraphSAGE](https://arxiv.org/abs/1706.02216) framework. In practice, in addition to reducing data size, sampling also enables data alignment, which facilitates efficient processing by Tensor-Based computing frameworks. <br />
<br />We abstract the user requirements for sampling into **2 classes** of operations:**NEIGHBORHOOD SAMPLING**, **NEGATIVE SAMPLING**. Neighbor sampling, depending on the input vertex, samples its one-hop or multi-hop neighbor vertices for constructing the perceptual field in [GCN](https://arxiv.org/abs/1609.02907) theory. The input of neighbor sampling can be from the output of graph traversal or from other data sources outside the graph. Negative sampling, based on the input vertices, samples vertices that are not directly connected to it. Negative sampling is often used as an important tool for supervised learning. <br />
<br />Each type of sampling operation has different implementation strategies, such as random, edge-weight, etc. Combined with practical production, we have accumulated **more than 10** sampling operators and opened the operator programming interface to allow user customization to meet the needs of the rapidly evolving GNN. This chapter introduces neighbor sampling, and negative sampling will be introduced in detail in the next chapter. In addition, subgraph sampling, proposed at the recent AI Top Meeting, is under development. <br

### Usage

#### Interface
The sampling operator takes the meta-path and the number of samples as input, and is used to express support for arbitrary heterogeneous graphs and arbitrary hop sampling. The sampling results are organized into `Layers` objects, and the result of each hop sampling is a `Layer` for which the corresponding `Nodes` and `Edges` can be obtained. A sampling operation can be implemented in 3 steps as follows.

- Define the sampling operator by `g.neighbor_sampler()` to get the `NeighborSampler` object `S`.
- Call `S.get(ids)`, to get the vertex's neighbor `Layers` object `L`.
- call `L.layer_nodes(i)`, `L.layer_edges(i)` to get the vertices and edges of the `i`th hop.



```python
def neighbor_sampler(meta_path, expand_factor, strategy="random"):
"""
Args:
  meta_path(list): string list consisting of edge_type, referring to the path sampled by the neighbor;
  expand_factor(list): list composed of int, the i-th element represents the number of neighbors sampled in the i-th hop; length must be the same as meta_path
  strategy(string): sampling strategy, see below for a detailed explanation
Return:
  NeighborSampler object
"""
```

```python
def NeighborSampler.get(ids):
""" sample the one-hop or multi-hop neighbors of the specified ids
Args:
  ids(numpy.ndarray): one-dimensional int64 array
Return:
  Layers object
"""
```

Sampling returns a `Layers` object, which means "the neighboring vertices of the source vertex, and the edges between the source vertex and the neighboring vertices". The shape of the ids of each `Layer` is two-dimensional, **[the size of the expansion of the previous layer of ids, the number of samples of the current layer]**.

```python
def Layers.layer_nodes(layer_id):
""" Get the `Nodes` of the ith layer layer, layer_id starts from 1. """
    
def Layers.layer_edges(layer_id):
""" Get the `Edges` of the ith layer, with layer_id starting at 1. """
```

<br />In GSL, refer to the `g.out*` and `g.in*` series interfaces. For example

```python
# Sample a one-hop neighbor vertex
g.V().outV().sample(count).by(strategy)

# Sample two-hop neighbor vertices
g.V().outV().sample(count).by(strategy).outV().sample(count).by(strategy)
```

#### Example
As shown in the figure below, starting from a vertex of type user, sample its 2-hop neighbors, and return the result as layers, which contains layer1 and layer2. **layer's index starts from 1**, i.e. 1-hop neighbor is layer1 and 2-hop neighbor is layer2.

![2-hop-sampling](../../../images/2-hop-sampling.png)

```python
s = g.neighbor_sampler(["buy", "i2i"], expand_factor=[2, 2])
l = s.get(ids) # input ids: shape=(batch_size)

# Nodes object
# shape=(batch_size, expand_factor[0])
l.layer_nodes(1).ids
l.layer_nodes(1).int_attrs

 # Edges object
 # shape=(batch_size * expand_factor[0], expand_factor[1])
l.layer_edges(2).weights
l.layer_edges(2).float_attrs
```

### Sampling Strategies
GL currently has support for the following sampling strategies, corresponding to the `strategy` parameters when generating `NeighborSampler` objects.

| **strategy** | **description** |
| --- | --- |
| edge_weight | Samples with probability with edge weights |
| random | random_with_replacement |
| topk | Return the neighbors with edge weight topK, if there are not enough neighbors, refer to the padding rule.
| in_degree | Probability sampling by vertex degree.
| full | Returns all neighbors, the expand_factor parameter does not work, the result object is SparseNodes or SparseEdges, see "[graph query](graph_query.md)" for the object description.


<br />Padding rules: When there is not enough data for the sampling request, the returned result needs to be filled in some way. By default, `default_neighbor_id` is used to fill in the insufficient `id`. `default_neighbor_id` defaults to 0 and can be set by `gl.set_default_neighbor_id(xx)`. To loop padding, i.e. to loop existing neighbor ids instead of `default_neighbor_id`, set the padding mode `gl.CIRCULAR`, `gl.set_padding_mode(gl.CIRCULAR)`.
