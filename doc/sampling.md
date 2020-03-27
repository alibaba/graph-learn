# Introduction

The sampling interface is described by `sample()` and `by()`.
The sampled neighbors can be nodes or edges, and the result is represented with **Nodes** and **Edges**.
Refer to [Values](../graphlearn/python/values.py) for details about the structure.

All sampling queries start from a node, and then use `outE()` and `outV()` to get the neighbor edges and nodes.
If the current query is on an edge, use `outV()` and `inV()` to switch to the src_node and dst_node first.
And then sample neighbors like on a node.


# Strategies

**GL** provides several sampling strategies to satisfy different **GNN** algorithms.
Currently, the built-in sampling strategies are: **random**, **in_degree**, **edge_weight** and **full**.
If you want to customize a kind of sampling strategy, please refer to [Operator](operator.md).

## Random

Randomly sample neighbor nodes or edges for the current nodes.

For exmaple, sample 2-hop neighbor nodes for user node along meta-path "user--buy-->item--similar_to-->item".
```python
res = g.V("user").batch(64)                         \ # now the query is on user nodes, with batch size of 64
 .outV("buy").sample(2).by("random").alias("a")     \ # now on item nodes, sample 2 of them for each of the previous users
 .outV("similar").sample(2).by("random").alias("b") \ # now on item, sample 2 of them for each of the previous items
 .emit(lambda x: (x["a"], x["b"]))
# res = (Nodes[64*2], Nodes[64*2*2])
```

## InDegree

Sample neighbor nodes or edges for the current nodes based on the in-degree distribution of neighbor nodes.

```python
q = g.V("user").batch(64)                              \ # now the query is on user nodes, with batch size of 64
     .outV("buy").sample(2).by("in_degree").alias("a") \ # now on item nodes, sample 2 of them for each of the previous users
     .values(lambda x: x["a"])
print(g.run(q))
```

## EdgeWeight

Sample neighbor nodes or edges for the current nodes based on the edge weight distribution.

```python
g.V("user").batch(64)                              \ # now the query is on user nodes, with batch size of 64
 .outE("buy").sample(2).by("edge_weight")          \ # now on edge buy, 1 hop sample by edge_weight
 .inV().alias("a")                                 \ # switch to the dst_node of edge buy, that is item
 .inV("similar").sample(3).by("random").alias("b") \ # 2 hop sample by random
 .emit()
```

## FULL

Get the full neighbor nodes or edges for the current ndoes.
Here the parameter of `sample()` can be empty.

It returns [SparseNodes](../graphlearn/python/values.py) or [SparseEdges](../graphlearn/python/values.py).

```python
g.V("user").batch(64)                        \ # now the query is on user nodes, with batch size of 64
 .outV("buy").sample().by("full").alias("a") \ # get all neighbor items for each of the previous users
 .emit(lambda x: x["a"])
```

[Home](../README.md)
