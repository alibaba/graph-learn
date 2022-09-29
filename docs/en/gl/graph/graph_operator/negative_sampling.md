## Negative sampling

### Introduction
Negative sampling, an important tool for unsupervised training, refers to sampling vertices that have no direct edge relationship with a given vertex. Similar to neighbor sampling, negative sampling has different implementation strategies, such as random, in-degree of nodes, etc. As a common operator of GNN, negative sampling supports extensions and scenario-oriented customization. <br /> 

### Usage

#### interface
The negative sampling operator takes an edge or vertex type as input. When edge type is given, it means "sample vertices not directly related to the given vertex at this edge type", and the candidate set is the vertices of the edge's destination that are not directly related to the given vertex. When the vertex type is given, it means "sample vertices of this type that are not directly associated with the given vertex", and the user needs to specify the set of candidate vertices. The sampling result is organized into `Nodes` objects (similar to one-hop neighbor sampling, but without the `Edges` objects). A negative sampling operation can be implemented in 3 specific steps as follows.

- Define the negative sampling operator by `g.negative_sampler()` to obtain the `NegativeSampler` object `S`.
- call `S.get(ids)`, to get the `Nodes` object.
- call the [interface](../graph_operator/graph_query.md) of the `Nodes` object to get the specific values.


```python
def negative_sampler(object_type, expand_factor, strategy="random"):
"""
Args:
  object_type(string): edge type or vertex type
  expand_factor(int): number of negative samples for each vertex
  strategy(string): sampling strategy, see below for details
Return:
  NegativeSampler object
"""
```

```python
def NegativeSampler.get(ids, **kwargs):
""" negatively sample the specified vertex ids
Args:
  ids(numpy.ndarray): one-dimensional int64 array
  **kwargs: extended arguments, different sampling strategies may require different arguments
Return:
  Nodes object
"""
```



#### Example

```python
es = g.edge_sampler("buy", batch_size=3, strategy="random")
ns = g.negative_sampler("buy", 5, strategy="random")

for i in range(5):
    edges = es.get()
    neg_nodes = ns.get(edges.src_ids)
    
    print(neg_nodes.ids) # shape is (3, 5)
    print(neg_nodes.int_attrs) # shape is (3, 5, count(int_attrs))
    print(neg_nodes.float_attrs) # shape as (3, 5, count(float_attrs))
```

### Negative sampling strategies
GL currently supports the following negative sampling strategies, corresponding to the `strategy` argument when generating `NegativeSampler` objects.

| **strategy** | **description** |
| --- | --- |
| random | Random negative sampling, not guaranteed true-negative |
| in_degree | Negative sampling with probability of vertex entry distribution, guaranteed true-negative |
| node_weight | Negative sampling with probability of vertex weight, true-negative |

## Negative sampling by specified attribute condition

GL provides the ability to negative sampling by a given attribute column, adding new parameters in g.negative_sampler and requiring the input to be a positive sample pair (src_ids, dst_ids). <br />

- Definition<br />

```python
def negative_sampler(object_type, expand_factor, strategy='random', 
                     conditional=True, # new parameters, all the following are new parameters (optional)
                     unique=False,
                     int_cols=[],
                     int_props=[],
                     float_cols=[],
                     float_props=[],
                     str_cols=[],
                     str_props=[]):
"""
Args:
    object_type(string): edge type or vertex type
    expand_factor(int): number of negative samples
    strategy(string): sampling strategy, supports random, in_degree, node_weight
    conditional(bool): whether to use conditional negative sampling. The value is set to True for conditional negative sampling
    unique(bool): if or not the negative samples need to be unique.
    int_cols(list): subscript of the specified int type attribute, indicating negative sampling under these specified attributes. For example, the positive samples of the input
        int attribute of dst_ids in this pair is 3. int_cols=[0,1] means that the first int attribute and the 1st
        int attribute of dst_ids, and the second node with the same int attribute as the second attribute of dst_ids, and select the negative samples.
    int_props(list): The proportion of each attribute sampled in int_cols. For example, int_cols=[0,1],int_props=[0.1,0.2],
        means that expand_factor*0.1 negative samples are sampled at the same point as the 1st int attribute of dst_ids, and at the same point as the 2nd int attribute of dst_ids
        The second int attribute of dst_ids is sampled at the same point as the second int attribute of dst_ids.
    float_cols(list): subscript of the specified float property, same as int_cols.
    float_props(list): proportion of each attribute of float_cols, same as int_props.
    str_cols(list): subscript of the specified string_properties, same as int_cols.
    str_props(list): the proportion of each attribute of str_cols, same as int_props.
Return:
    NegativeSampler object
"""
```

**Note:**<br />
When negative sampling, it will follow the strategy specified by strategy in the specified property condition to negatively sample, requiring sum(int_props) + sum(float_props) + sum(str_props) <= 1. If the value < 1, the remaining negative samples will no longer be sampled according to the specified property condition, but only according to strategy.

- Interfaces<br />

```python
def get(src_ids, dst_ids):
""" Negative sampling of the specified src_ids, dst_ids positive sample pairs.
Args:
    src_ids(numpy.ndarray): one-dimensional int64 array, ids of positive sample source nodes
    dst_ids(numpy.ndarray): 1-dimensional int64 array, ids of positive sample destination nodes
Return:
    Nodes object
"""
```

Sampling will remove all neighbors of all src_ids. <br />

- Example<br />

```python
"""
Suppose the point type is item and it has 3 int attributes, 1 float attribute, and 1 string attribute.
The positive sample is :
    src_ids = np.array([1,2,3,4,5])
    dst_ids = np.array([6,2,3,5,9])
Now you need to negatively sample the points from the given point table according to the 'node_weight' strategy and require that the points with the 1st int attribute value equal to dst_ids
and sampling 2 negative nodes at the point where the value of the 1st int attribute is equal to the value of the 1st string attribute of dst_ids
2 negative samples
"""
s = g.negative_sampler('item',
                       expand_factor=4,
                       strategy='node_weight',
                       conditional=True,
                       unique=False,
                       int_cols=[0],
                       int_props=[0.5],
                       str_cols=[0],
                       str_props=[0.5])
src_ids = np.array([1,2,3,4,5])
dst_ids = np.array([6,2,3,5,9])
nodes = s.get(src_ids, dst_ids)
```
