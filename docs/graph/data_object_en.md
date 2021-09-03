# Data objects

GraphLearn describes the result of traversal and sampling as a data object; GraphLearn traversal and sampling are Batch operations, in which the number of neighbors/negative neighbors of a Batch can be equal or unequal, so sampling is divided into aligned and non-aligned sampling.

The result of vertex traversal and aligned vertex sampling is `Nodes`, and the result of non-aligned vertex sampling is `SparseNodes`. Correspondingly, edge traversal and aligned edge sampling results in `Edges`, and non-aligned edge sampling results in `SparseEdges`. <br />

## Dense data objects
### `Nodes`

```python
@property
def ids(self):
""" vertex id, numpy.ndarray(int64) """

@property
def shape(self):
""" vertex id's shape, (batch_size) / (batch_size, neighbor_count) """

@property
def int_attrs(self):
""" attributes of type int, numpy.ndarray(int64), shape as [ids.shape, number of attributes of type int] """

@property
def float_attrs(self):
""" properties of type float, numpy.ndarray(float32), shape is [ids.shape, number of properties of type float] """

@property
def string_attrs(self):
""" attributes of type string, numpy.ndarray(string), shape is [ids.shape, number of attributes of type string] """

@property
def weights(self):
""" weights, numpy.ndarray(float32), shape is ids.shape """

@property
def labels(self):
""" labels, numpy.ndarray(int32), shape is ids.shape """ @property
def ids(self):
""" vertex id, numpy.ndarray(int64) """

@property
def shape(self):
""" vertex id's shape, (batch_size) / (batch_size, neighbor_count) """

@property
def int_attrs(self):
""" attributes of type int, numpy.ndarray(int64), shape as [ids.shape, number of attributes of type int] """

@property
def float_attrs(self):
""" properties of type float, numpy.ndarray(float32), shape is [ids.shape, number of properties of type float] """

@property
def string_attrs(self):
""" attributes of type string, numpy.ndarray(string), shape is [ids.shape, number of attributes of type string] """

@property
def weights(self):
""" weights, numpy.ndarray(float32), shape is ids.shape """

@property
def labels(self):
""" labels, numpy.ndarray(int32), shape as ids.shape """
```

### `Edges`
The difference between the `Edges` interface and `Nodes` is that the `ids` interface has been removed and the following four interfaces have been added for accessing source and destination vertices.

```python
@property
def src_nodes(self):
""" source vertex Nodes object """

@property
def dst_nodes(self):
""" destination vertex Nodes object """

@property
def src_ids(self):
""" source vertex id, numpy.ndarray(int64) """

@property
def dst_ids(self):
""" destination vertex id, numpy.ndarray(int64) """
```

Regarding the shape of `ids`, in vertex and edge traversal operations, the shape is one-dimensional and the size is the specified batch size. In sampling operations, the shape is two-dimensional and the size is [the one-dimensional expansion size of the input data, the current number of samples].

## Sparse data object

### `SparseNodes`
`SparseNodes` is used to express the sparse neighbor vertices of a vertex, with the following additional interface relative to Nodes.

```python
@property
def offsets(self):
""" one-dimensional shape-shifting array: the number of neighbors per vertex """

@property
def dense_shape(self):
""" tuples with 2 elements: the shape of the corresponding Dense Nodes """

@property
def indices(self):
""" 2-dimensional array representing the position of each neighbor """

def __next__(self):
""" the traversal interface, traversing the vertices of each vertex's neighbors """
  return Nodes
```

### `SparseEdges`
``SparseEdges`` is used to express the sparse neighboring edges of a vertex, with the following additional interface relative to Edges.

```python
@property
def offsets(self):
""" one-dimensional shape-shifting array: the number of neighbors per vertex """

@property
def dense_shape(self):
""" tuples with 2 elements: the shape of the corresponding Dense Edges """

@property
def indices(self):
""" 2-dimensional array representing the position of each neighbor """

def __next__(self):
""" the traversal interface, traversing the edges of each vertex's neighbors """
  return Edges
```