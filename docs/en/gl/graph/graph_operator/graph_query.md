## Graph query

After the graph object is constructed, graph query operation can be performed. Querying means getting **meta information** and **data information** about the graph, without involving complex computation and sampling logic. <br />

### Meta

Meta-information refers to information about the graph structure and statistical types, including the topology of the graph, the total number of vertices, the distribution of edges and vertices, the maximum discrepancy of vertices, etc. <br />

#### Topology <br />

```python
def get_topology()
""" Get the topology of the graph
Return type is dict, where key is edge_type and value contains src_type, dst_type attributes.
"""
```

<br />The heterogeneous graph shown below, get its topology and return the result format see the sample code.

![topology](../../../../images/topology.png)


<br /> Figure 1: Topology information of the graph

```python
g = Graph(...)
g.init(...)
topo = g.get_topology()
topo.print_all()

"""
egde_type:buy, src_type:user, dst_type:item
egde_type:click, src_type:user, dst_type:item
egde_type:swing, src_type:item, dst_type:item
"""
```


#### InDegree and OutDegree

To be updated. <br />

#### Graph Statistics
```
# get the number of nodes and edges on each server.
g = Graph(...)
g.init(...)
# client method
g.get_stats()

# server method
g.server_get_stats()
```

### Data

**GL** has two basic data types: `Nodes` and `Edges`. The return object of traversal, query, and sampling operations is the vertex or edge of a batch. In particular, non-aligned sampling returns sparse forms of the two basic data types, `SparseNodes` and `SparseEdges`. <br />

### Vertex queries

`Nodes` can be derived from graph traversal or sampling, or the vid can be specified directly. Regardless of the source, their attributes, weights, or labels can be queried. <br

1. specify vid to query vertices<br />

```python
def get_nodes(node_type, ids)
''' Get the weights, labels, and attributes of vertices of the specified type
Args:
    node_type(string): vertex type
    ids(numpy.array): vertex id
Return:
    Nodes object
'''
```

<br />The following data demonstrates the usage of the `get_nodes()` interface.

Table 1: user vertex data

| id | attributes |
| --- | --- |
| 10001 | 0:0.1:0 |
| 10002 | 1:0.2:3 |
| 10003 | 3:0.3:4 |

```python
g = Graph(...)
u_nodes = g.get_nodes("user", np.array([10001, 10002, 10003]))

print(u_nodes.int_attrs) # shape = [3, 2]
# array([[0, 0], [1, 3], [2, 4]])

print(u_nodes.float_attrs) # shape = [3, 1]
# array([[ 0.1], [0.2], [0.3]])
```

2. Traversing vertex queries <br />

``` python
sampler = g.node_sampler("user", batch_size=3, strategy="random")
for i in range(5):
    nodes = sampler.get()
    print(nodes.ids) # shape: (3, )
    print(nodes.int_attrs) # shape: (3, int_attr_num)
    print(nodes.float_attrs) # shape: (3, float_attr_num)

```

3. sampled vertex attribute query<br />

```python
s = g.neighbor_sampler(["buy", "i2i"], expand_factor=[3, 2])
l = s.get(ids) # input ids: shape=(batch_size)

# 1-hop Nodes object
l.layer_nodes(1).ids # shape=(batch_size, 3)
l.layer_nodes(1).int_attrs # shape=(batch_size, 3, int_attr_num)

# 2-hop Nodes object
l.layer_nodes(2).ids # shape=(batch_size * 3, 2)
l.layer_nodes(2).int_attrs # shape=(batch_size * 3, 2, int_attr_num)
```

### Edge queries<br />

`Edges` can be traversed or sampled from the graph. Edges that are traversed or sampled can be queried for their attributes, weights, or labels.
<br />

- Traversed Edges Query<br />

```python
sampler = g.edge_sampler("buy", batch_size=3, strategy="random")
for i in range(5):
    edges = sampler.get()
    print(edges.src_ids) # shape: (3, )
    print(edges.dst_ids) # shape: (3, )
    print(edges.weights) # shape: (3, )
    print(edges.float_attrs) # shape: (3, float_attr_num)
```

- Sampled edge attributes query <br />

```python
s = g.neighbor_sampler(["buy", "i2i"], expand_factor=[3, 2])
l = s.get(ids) # input ids: shape=(batch_size)

# 1 hop Edges object, i.e. buyi edge
l.layer_edges(1).weights # shape=(batch_size, 3)
l.layer_edges(1).float_attrs # shape=(batch_size, 3, float_attr_num)

# 2-hop Edges object, i.e. i2i edges
l.layer_edges(2).weights # shape=(batch_size * 3, 2)
l.layer_edges(2).float_attrs # shape=(batch_size * 3, 2, float_attr_num)
```

### Sparse vertex/edge

<br />
The result of traversal, sampling is generally a `Nodes`/`Edges` object, which can be queried using the interface above. <br /> In non-aligned sampling, the result is sparse. For example, in full-neighbor sampling (i.e., neighbor sampling with the "full" strategy), the neighbors are not aligned because the number of neighbors per vertex is not fxied-size. <br /> <br /> In the following, we use the "full" strategy as an example.
<br />In the following, we illustrate the usage of the interface for sparse objects, using the edge property query with full neighbor sampling as an example. <br />

Table 4: buy edge data<br />

| user | item | weight |
| --- | --- | --- |
| 1 | 3 | 0.2 |
| 1 | 0 | 0.1 |
| 1 | 2 | 0.0 | 0.1
| 2 | 1 | 0.1 | ---
| 4 | 1 | 0.5 |
| 4 | 2 | 0.3 |


```python
s = g.neighbor_sampler("buy", expand_factor=0, strategy="full")
l = s.get(ids) # input ids: shape=(4)

# res['a'] # Nodes of [1, 2, 3, 4]
# res['b'] # SparseEdges

nodes = l.layer_nodes(1)
edges = l.layer_edges(1)

nodes.ids 
# array([3, 0, 2, 1, 1, 2])

edges.src_ids 
# array([1, 1, 1, 2, 4, 4])
edges.dst_ids
# array([3, 0, 2, 1, 1, 2])

nodes.offsets 
# [3, 1, 0, 2] 
# i.e. user1 has 3 neighbors, user2 has 1 neighbor, user3 has 0 neighbors, and user4 has 2 neighbors
edges.offsets 
# [3, 1, 0, 2]

nodes.dense_shape
# [4, 3]
# i.e. [number of seed vertices, maximum number of neighbors in seed vertices]
edges.dense_shape
# [4, 3]

nodes.indices
# [[0, 1], [0, 2], [0, 3], [1, 0], [3, 1], [3, 2]]
# The corresponding subscripts of src_ids in dense Nodes (same as dst_ids in dense Nodes)
# Corresponding dst dense Nodes.
# [[ 3, 0, 2],
# [ 1, -1, -1],
# [ -1, -1, -1 ],
# [ [1, 2, -1]]
edges.indices
# [[0, 1], [0, 2], [0, 3], [1, 0], [3, 1], [3, 2]]

edges.weights
# [0.2, 0.1, 0.0, 0.1, 0.5, 0.3]

# Iterate over all neighboring edges of each input vertex ids.
iterate = 0
for e in edges:
    print("Iterate {}:".format(iterate), e.dst_ids, e.weights)
    iterate += 1
# Iterate 0: [3, 0, 2], [0.2, 0.1, 0.0]
# Iterate 1: [1], [0.1],
# Iterate 2: [], []
# Iterate 3: [1, 2], [0.5, 0.3]

# Iterate over all neighboring vertices of each input vertex ids.
iterate = 0
for n in nodes:
    print("Iterate {}:".format(iterate), n.ids)
    iterate += 1
# Iterate 0: [3, 0, 2]
# Iterate 1: [1]
# Iterate 2: []
# Iterate 3: [1, 2]
```

### Default value setting

For points and edges that do not exist in the graph, the query will return the default value:<br />
label, with a default value of -1<br />
int_attrs, default value is 0<br />
float_attrs, default value is 0.0<br />
string_attrs, the default value is ''<br />
The default values of the int, float, and string attributes can be set via global system parameters. <br />
