# 数据对象

GraphLearn将遍历、采样的结果描述为数据对象。GraphLearn的遍历和采样都为Batch操作，在采样中，一个Batch的邻居/负邻居数可以相等也可以不相等，因此采样又分为对齐采样和非对齐采样。

顶点的遍历和对齐的顶点采样结果为`Nodes`，非对齐的顶点采样结果为`SparseNodes`。相应地，边的遍历和对齐的边采样结果为`Edges`，非对齐的边采样结果为`SparseEdges`。<br />

## Dense数据对象
### `Nodes`

```python
@property
def ids(self):
""" 顶点id，numpy.ndarray(int64) """

@property
def shape(self):
""" 顶点id的shape, (batch_size) / (batch_size, neighbor_count) """

@property
def int_attrs(self):
""" int类型的属性，numpy.ndarray(int64)，shape为[ids.shape, int类型属性的个数] """

@property
def float_attrs(self):
""" float类型的属性，numpy.ndarray(float32)，shape为[ids.shape, float类型属性的个数] """

@property
def string_attrs(self):
""" string类型的属性，numpy.ndarray(string)，shape为[ids.shape, string类型属性的个数] """

@property
def weights(self):
""" 权重，numpy.ndarray(float32)，shape为ids.shape """

@property
def labels(self):
""" 标签，numpy.ndarray(int32)，shape为ids.shape """@property
def ids(self):
""" 顶点id，numpy.ndarray(int64) """

@property
def shape(self):
""" 顶点id的shape, (batch_size) / (batch_size, neighbor_count) """

@property
def int_attrs(self):
""" int类型的属性，numpy.ndarray(int64)，shape为[ids.shape, int类型属性的个数] """

@property
def float_attrs(self):
""" float类型的属性，numpy.ndarray(float32)，shape为[ids.shape, float类型属性的个数] """

@property
def string_attrs(self):
""" string类型的属性，numpy.ndarray(string)，shape为[ids.shape, string类型属性的个数] """

@property
def weights(self):
""" 权重，numpy.ndarray(float32)，shape为ids.shape """

@property
def labels(self):
""" 标签，numpy.ndarray(int32)，shape为ids.shape """
```

### `Edges`
`Edges`接口与`Nodes`的区别为，去掉了`ids`接口，增加了以下4个接口，用于访问源顶点和目的顶点。

```python
@property
def src_nodes(self):
""" 源顶点Nodes对象 """

@property
def dst_nodes(self):
""" 目的顶点Nodes对象 """

@property
def src_ids(self):
""" 源顶点id，numpy.ndarray(int64) """

@property
def dst_ids(self):
""" 目的顶点id，numpy.ndarray(int64) """
```

关于`ids`的shape，在顶点和边遍历操作中，shape为一维，大小为指定的batch size。在采样操作中，shape为二维，大小为[输入数据的一维展开大小，当前采样个数]。

## Sparse数据对象

### `SparseNodes`
`SparseNodes`用于表达顶点的稀疏邻居顶点，相对于Nodes增加了以下接口。

```python
@property
def offsets(self):
""" 一维整形数组: 每个顶点的邻居个数 """

@property
def dense_shape(self):
""" 含有2个元素的tuple: 对应的Dense Nodes的shape """

@property
def indices(self):
""" 二维数组，代表每一个邻居的位置 """

def __next__(self):
""" 遍历接口，遍历每个顶点的邻居顶点们 """
  return Nodes
```

### `SparseEdges`
`SparseEdges`用于表达顶点的稀疏邻边，相对于Edges增加了以下接口。

```python
@property
def offsets(self):
""" 一维整形数组: 每个顶点的邻居个数 """

@property
def dense_shape(self):
""" 含有2个元素的tuple: 对应的Dense Edges的shape """

@property
def indices(self):
""" 二维数组，代表每一个邻居的位置 """

def __next__(self):
""" 遍历接口，遍历每个顶点的邻居边们 """
  return Edges
```