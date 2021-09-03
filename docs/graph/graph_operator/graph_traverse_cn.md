## 图遍历

<a name="pLeth"></a>
### 介绍
图遍历，在GNN里的语义有别于经典的图计算。主流深度学习算法的训练模式会按batch迭代。为了满足这种要求，数据要能够按batch访问，我们把这种数据的访问模式称为遍历。在GNN算法中，数据源为图，训练样本通常由图的顶点和边构成。图遍历是指为算法提供按batch获取顶点、边或子图的能力。

目前**GL**支持顶点和边的batch遍历。这种随机遍历可以是无放回的，也可以是有放回的。在无放回遍历中，每当一个epoch结束后都会触发`gl.OutOfRangeError`。被遍历的数据源是划分后的，即当前worker（以分布式TF为例）只遍历与其对应的Server上的数据。

<a name="Fj1gp"></a>
### 顶点遍历
<a name="HEDng"></a>
#### 用法
顶点的数据来源有3种：所有unique的顶点，所有边的源顶点，所有边的目的顶点。顶点遍历依托`NodeSampler`算子实现，Graph对象的`node_sampler()`接口返回一个`NodeSampler`对象，再调用该对象的`get()`接口返回`Nodes`格式的数据。

```python
def node_sampler(type, batch_size=64, strategy="by_order", node_from=gl.NODE):
"""
Args:
  type(string):     当node_from为gl.NODE时，为顶点类型，否则为边类型;
  batch_size(int):  每次遍历的顶点数
  strategy(string): 可选值为"by_order"和"random"，表示无放回遍历和随机遍历。当为"by_order"时，若触底后不足batch_size，则返回实际数量，若实际数量为0，则触发gl.OutOfRangeError
  node_from:        数据来源，可选值为gl.NODE、gl.EDGE_SRC、gl.EDGE_DST;
Return:
  NodeSampler对象
"""
```


```python
def NodeSampler.get():
"""
Return:
    Nodes对象，若非触底，预期ids的shape为[batch_size]
"""
```

<br />通过`Nodes`对象获取具体的值，如id、weight、attribute等，参考[API](graph_query_cn.md)。在GSL中，顶点遍历参考`g.V()`。<br />

<a name="aNB50"></a>
#### 示例

user顶点表：<br />

| id | attributes |
| --- | --- |
| 10001 | 0:0.1:0 |
| 10002 | 1:0.2:3 |
| 10003 | 3:0.3:4 |


buy边表：<br />

| src_id | dst_id  | attributes |
| --- | --- | --- |
| 10001 | 1 | 0.1 |
| 10001 | 2 | 0.2 |
| 10001 | 3 | 0.4 |
| 10002 | 1 | 0.1 |


```python
# Exmaple1: 随机采样顶点。
sampler1 = g.node_sampler("user", batch_size=3, strategy="random")
for i in range(5):
  nodes = sampler1.get()
  print(nodes.ids) # shape=(3, )
  print(nodes.int_attrs) # shape=(3, 2)，有2个int属性
  print(nodes.float_attrs) # shape=(3, 1)，有1个float属性

# Exmaple2: 遍历图中的user顶点
sampler2 = g.node_sampler("user", batch_size=3, strategy="by_order")
while True:
  try:
    nodes = sampler1.get()
    print(nodes.ids) # 除最后一个batch外，shape为(3, )，最后一个batch的shape为剩余的id数
    print(nodes.int_attrs)
    print(nodes.float_attrs)
  except gl.OutOfRangError:
    break

# Exmaple3: 遍历图中的buy边的源顶点，即user顶点，为unique的
sampler2 = g.node_sampler("user", batch_size=3, strategy="by_order", node_from=gl.EDGE_SRC)
while True:
  try:
    nodes = sampler1.get()
    print(nodes.ids) # shape=(2, )，由于buy边表中src_id只有2个unique的值，不满batch_size 3，因此这个循环只进行了一次
    print(nodes.int_attrs)
    print(nodes.float_attrs)
  except gl.OutOfRangError:
    break
```


<a name="8lRI5"></a>
### 边遍历
<a name="EWBuj"></a>
#### 用法
边遍历依托`EdgeSampler`算子实现。Graph对象的`edge_sampler()`接口返回一个`EdgeSampler`对象，再调用该对象的`get()`接口返回`Edges`格式的数据。

```python
def edge_sampler(edge_type, batch_size=64, strategy="by_order"):
"""
Args:
  edge_type(string): 边类型
  batch_size(int):   每次遍历的边数
  strategy(string):  可选值为"by_order"和"random"，表示无放回遍历和随机遍历。当为"by_order"时，若触底后不足batch_size，则返回实际数量，若实际数量为0，则触发gl.OutOfRangeError
Return:
  EdgeSampler对象
"""
```

```python
def EdgeSampler.get():
"""
Return:
    Edges对象，若非触底，预期src_ids的shape为[batch_size]
"""
```

<br />通过`Edges`对象获取具体的值，如id、weight、attribute等，参考[API](graph_query_cn.md#FPU74)。在GSL中，边遍历参考`g.E()`。<br />

<a name="RVPmZ"></a>
#### 示例<br />

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

