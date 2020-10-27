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





