# Graph Traversal

<a name="pLeth"></a>
# 1. Introduction
The semantics of graph traversal in GNN is different from classic graph computation. Training by batch is adopted by mainstream deep learning alogrithms and to meet this requirement, graph data must be able to access in batches. We call this data access pattern graph traversal. In a GNN algorithm, the data source is the graph  which mostly are samples of vertices and edges and graph traversal refers to the approach to obtain vertices and edges in batch.

Currently **GL** supports traversing graph by batch and the strategy could be random or without replacement. The traversal without replacement will throw a `gl.OutOfRangeError` exception when an epoch is finished. The data source to be traversed is partitioned, which means the worker(e.g. distributed tensorflow) only traverses the data on the corresponding Server.

<a name="Fj1gp"></a>
# 2. Vertex Traversal
<a name="HEDng"></a>
## 2.1 Usages
There are three type of data sources for vertices: all unique vertices, source vertices of all edges and destination vertices of all edges. Vertex traversal is implemented by the `NodeSampler` operator. The `node_sampler()` API of `Graph` object returns a `NodeSampler` object, and invoking `get()` API of this object returns `Nodes` data.

```python
def node_sampler(type, batch_size=64, strategy="by_order", node_from=gl.NODE):
"""
Args:
  type(string):     type="vertex" when node_from=gl.NODE, otherwise type="edge";
  batch_size(int):  number of vertices in each traversal
  strategy(string):  can only take value from {"by_order, "random"}.
      "by_order": stands for traverse without replacements, the return number 
          is the actual traversed count, and a gl.OutOfRangeError will throw if 
          the actual traversed count is 0; 
      "random": stands for random traverse.
  node_from: data source, can only take value from {gl.NODE、gl.EDGE_SRC、gl.EDGE_DST}
Return:
  NodeSampler object
"""
```

```python
def NodeSampler.get():
"""
Return:
    Nodes object, data shape=[batch_size] if current epoch isn't finished.
"""
```

<br />You can access specific values through the `Nodes` object, such as id, weight, attribute, etc. See APIs for reference. And refer to `g.V()` for vertex traversal in GSL.<br />

<a name="aNB50"></a>
## 2.2 Example
| id | attributes |
| --- | --- |
| 10001 | 0:0.1:0 |
| 10002 | 1:0.2:3 |
| 10003 | 3:0.3:4 |

```python
sampler = g.node_sampler("user", batch_size=3, strategy="random")
for i in range(5):
    nodes = sampler.get()
    print(nodes.ids)
    print(nodes.int_attrs)
    print(nodes.float_attrs)
```

<a name="8lRI5"></a>
# 3 Edge Traversal
<a name="EWBuj"></a>
## 3.1 Usages
 Edge traversal is implemented by the `EdgeSampler` operator. The `edge_sampler()` API of `Graph` object returns a `EdgeSampler` object, and invoking `get()` API of this object returns `Edges` data.
 ```python
def edge_sampler(edge_type, batch_size=64, strategy="by_order"):
"""
Args:
  edge_type(string): edge type
  batch_size(int):   number of edges in each traversal
  strategy(string):  can only take value from {"by_order, "random"}.
      "by_order": stands for traverse without replacements, the return number 
          is the actual traversed count, and a gl.OutOfRangeError will throw if 
          the actual traversed count is 0; 
      "random": stands for random traverse.
Return:
  EdgeSampler object
"""
```
```python
def EdgeSampler.get():
"""
Return:
    Edges object, data shape=[batch_size] if current epoch isn't finished.
"""
```
<br />You can access specific values through the `Edges` object, such as id, weight, attribute, etc. See APIs for reference. And refer to `g.E()` for vertex traversal in GSL.<br />

<a name="RVPmZ"></a>
## 3.2 Example
| src_id | dst_id | weight | attributes |
| --- | --- | --- | --- |
| 20001 | 30001 | 0.1 | 0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19 |
| 20001 | 30003 | 0.2 | 0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29 |
| 20003 | 30001 | 0.3 | 0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39 |
| 20004 | 30002 | 0.4 | 0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49 |

```python
sampler = g.edge_sampler("buy", batch_size=3, strategy="random")
for i in range(5):
    edges = sampler.get()
    print(edges.src_ids)
    print(edges.src_ids)
    print(edges.weights)
    print(edges.float_attrs)
```

<br />
