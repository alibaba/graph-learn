## Graph traversal

### Introduction
Graph traversal, in GNN, has a different semantics than classical graph computation. The training model of mainstream deep learning algorithms iterates by batch. To meet this requirement, the data has to be accessible by batch, and we call this data access pattern traversal. In GNN algorithms, the data source is the graph, and the training samples usually consist of the vertices and edges of the graph. Graph traversal refers to providing the algorithm with the ability to access vertices, edges or subgraphs by batch.

Currently **GL** supports batch traversal of vertices and edges. This random traversal can be either putback-free or putback. In a no-replay traversal, `gl.OutOfRangeError` is triggered every time an epoch ends. The data source being traversed is partitioned, i.e. the current worker (in the case of distributed TF) only traverses the data on the Server corresponding to it.

### Vertex traversal

#### Usage
There are 3 sources of data for vertices: all vertices of uniqueness, source vertices of all edges, and destination vertices of all edges. Vertex traversal relies on the `NodeSampler` operator. The `node_sampler()` interface of the Graph object returns a `NodeSampler` object, which in turn calls the `get()` interface to return data in `Nodes` format.

```python
def node_sampler(type, batch_size=64, strategy="by_order", node_from=gl.NODE):
"""
Args:
  type(string): vertex type when node_from is gl.NODE, otherwise it is edge type;
  batch_size(int): the number of vertices to be traversed each time
  strategy(string): optional values are "by_order" and "random", which means ordered traversal and random traversal. When use "by_order", if the bottom is less than batch_size, the actual number will be returned, if the actual number is 0, gl.OutOfRangeError will be triggered.
  node_from: data source, optional values are gl;
Return:
  NodeSampler object
"""
```


```python
def NodeSampler.get():
"""
Return:
    Nodes object, if not bottomed out, expects the shape of ids to be [batch_size]
"""
```

<br />Get specific values such as id, weight, attribute, etc. from `Nodes` object, refer to [API](graph_query_en.md). In GSL, vertex traversal reference `g.V()`. <br />

#### Example

"user" vertex table:<br />

| id | attributes |
| --- | --- | 
| 10001 | 0:0.1:0 |
| 10002 | 1:0.2:3 |
| 10003 | 3:0.3:4 |


"buy" edge table:<br />

| src_id | dst_id | attributes |
| --- | --- | --- |
| 10001 | 1 | 0.1 |
| 10001 | 2 | 0.2 |
| 10001 | 3 | 0.4 |
| 10002 | 1 | 0.1 |


```python
# Exmaple1: Randomly sample vertices.
sampler1 = g.node_sampler("user", batch_size=3, strategy="random")
for i in range(5):
  nodes = sampler1.get()
  print(nodes.ids) # shape=(3, )
  print(nodes.int_attrs) # shape=(3, 2), with 2 int attributes
  print(nodes.float_attrs) # shape=(3, 1), with 1 float attribute

# Exmaple2: iterate over the user vertices in the graph
sampler2 = g.node_sampler("user", batch_size=3, strategy="by_order")
while True:
  try:
    nodes = sampler1.get()
    print(nodes.ids) # except for the last batch, the shape is (3, ), the shape of the last batch is the number of remaining ids
    print(nodes.int_attrs)
    print(nodes.float_attrs)
  except gl.OutOfRangError:
    break

# Exmaple3: Iterate over the source vertices of the buy edges of the graph, i.e. the user vertices, for the unique
sampler2 = g.node_sampler("user", batch_size=3, strategy="by_order", node_from=gl.EDGE_SRC)
while True:
  try:
    nodes = sampler1.get()
    print(nodes.ids) # shape=(2, ), because buy side table src_id only 2 unique values, dissatisfaction batch_size 3, so this loop is only carried out once
    print(nodes.int_attrs)
    print(nodes.float_attrs)
  except gl.OutOfRangError:
    break
```


### Edge traversal

#### Usage
Edge traversal relies on the `EdgeSampler` operator, and the `edge_sampler()` interface of the Graph object returns an `EdgeSampler` object, which in turn calls the `get()` interface to return data in `Edges` format.

```python
def edge_sampler(edge_type, batch_size=64, strategy="by_order"):
"""
Args:
  edge_type(string): edge type
  batch_size(int): number of edges per traversal
  strategy(string): optional values are "by_order" and "random", which means ordered traversal and random traversal. When use "by_order", if the bottom is less than batch_size, the actual number will be returned, if the actual number is 0, gl.OutOfRangeError will be triggered.
Return:
  EdgeSampler object
"""
```

```python
def EdgeSampler.get():
"""
Return:
    Edges object, if not bottomed, expects src_ids to have a shape of [batch_size]
"""
```

<br />Get specific values such as id, weight, attribute, etc. from `Edges` object, refer to [API](graph_query_en.md). In GSL, the edge traversal reference is `g.E()`. <br />

#### Example<br />

| src_id | dst_id | weight | attributes |
| --- | --- | --- | --- |
| 20001 | 30001 | 0.1 | 0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19 |
| 20001 | 30003 | 0.2 | 0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29 |
| 20003 | 30001 | 0.3 | 0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39 |
| 20004 | 30002 | 0.4 | 0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49 |

<br />

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
