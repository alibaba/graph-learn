# Introduction

In **GL**, negative smapling is treated as a kind of sampling.
It shares the same interface with [**neighbor sampling**](sampling.md).

All negative sampling queries start from a node, and then use `outNeg()` to sample negative neighbor nodes.
If the current query is on an edge, use `outV()` and `inV()` to switch to the src_node and dst_node first.

# Strategies

**GL** provides several negative sampling strategies to satisfy different **GNN** algorithms.
Currently, the built-in negative sampling strategies are: **random** and **in_degree**.
If you want to customize a kind of sampling strategy, please refer to [Operator](operator.md).

## Random

Randomly sample the negative neighbor nodes for the current nodes.

For exmaple, sample negative items for user nodes and then sample neighbor nodes for each sampled items.
It means get the similar items of the items that users did not buy.

```python
res = g.V("user").batch(64)                         \ # now we are on user nodes, with batch size of 64
 .outNeg("buy").sample(2).by("random").alias("a")   \ # now on item nodes, sample 2 of them for each of the previous user
 .outV("similar").sample(2).by("random").alias("b") \ # now on item, sample 2 of them for each of the previous item
 .emit(lambda x: (x["a"], x["b"]))
# res = (Nodes[64*2], Nodes[64*2*2])
```

## InDegree

Sample the negative neighbor nodes for the current nodes based on the in-degree distribution of negative nodes.

```python
q = g.V("user").batch(64)                                \ # now we are on user nodes, with batch size of 64
     .outNeg("buy").sample(2).by("in_degree").alias("a") \ # now on item nodes, sample 2 of them for each of the previous user
     .values(lambda x: x["a"].ids)
print(g.run(q))
```

[Home](../README.md)
