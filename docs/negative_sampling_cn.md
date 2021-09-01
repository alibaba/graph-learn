# 负采样

<a name="znmkl"></a>
# 1. 介绍
负采样作为无监督训练的重要手段，是指采样与给定顶点没有直接边关系的顶点。同邻居采样类似，负采样也有不同的实现策略，如随机、按顶点入度等。作为GNN的一种常见算子，负采样支持扩展和面向场景的自定义。<br />

<a name="6WYEX"></a>
# 2. 用法
<a name="keU3J"></a>
## 2.1 接口
负采样算子以边或顶点类型作为输入。当输入边类型时，表示“在该边类型下采样与给定顶点无直接关联的顶点”，候选集为边的目的顶点中与给定顶点不直接相连的顶点。当输入顶点类型时，表示“在该类型的顶点中采样与给定顶点无关联的顶点”，此时需要用户指定候选顶点集。采样结果组织成`Nodes`对象（类似一跳邻居采样，但不存在`Edges`对象）。一个负采样操作可具体分为以下3步实现：

- 通过`g.negative_sampler()`定义负采样算子，得到`NegativeSampler`对象`S`；
- 调用`S.get(ids)`, 得到`Nodes`对象；
- 调用`Nodes`对象的[接口](graph_query_cn.md#FPU74)获取具体的值；



```python
def negative_sampler(object_type, expand_factor, strategy="random"):
"""
Args:
  object_type(string): 边类型或顶点类型
  expand_factor(int):  负采样个数
  strategy(string):    采样策略，具体参考下文的详细解释
Return:
  NegativeSampler对象
"""
```

```python
def NegativeSampler.get(ids, **kwargs):
""" 对指定顶点ids进行负采样
Args:
  ids(numpy.ndarray): 一维int64数组
  **kwargs: 扩展参数，不同采样策略需要的参数可能不同
Return:
  Nodes对象
"""
```


<a name="B3CYq"></a>
## 2.2 示例

```python
es = g.edge_sampler("buy", batch_size=3, strategy="random")
ns = g.negative_sampler("buy", 5, strategy="random")

for i in range(5):
    edges = es.get()
    neg_nodes = ns.get(edges.src_ids)
    
    print(neg_nodes.ids)          # shape为(3, 5)
    print(neg_nodes.int_attrs)    # shape为(3, 5, count(int_attrs))
    print(neg_nodes.float_attrs)  # shape为(3, 5, count(float_attrs))
```

```python
# 1. 负采样一跳邻居顶点
g.V().outNeg(edge_type).sample(count).by(strategy)

# 负采样二跳邻居顶点
g.V().outNeg(edge_type).sample(count).by(strategy).outNeg(edge_type).sample(count).by(strategy)

# 在给定顶点候选集上负采样
g.V().Neg(node_type).sample(count).by("node_weight")
```


<a name="ePTLM"></a>
# 3 负采样策略
GL目前已支持以下几种负采样策略，对应产生`NegativeSampler`对象时的`strategy`参数。

| **strategy** | **说明** |
| --- | --- |
| random | 随机负采样，不保证true-negative |
| in_degree | 以顶点入度分布为概率进行负采样，保证true-negative |
| node_weight | 以顶点权重为概率进行负采样样，保证true-negative |

## 按指定属性条件的负采样

GL提供了按照给定的属性列来进行负采样的功能，在g.negative_sampler 里新增参数，并且要求输入为正样本对(src_ids, dst_ids)。<br />

- 定义<br />

```python
def negative_sampler(object_type, expand_factor, strategy='random', 
                     conditional=True, #新增参数，下面均为新增参数(可选)
                     unique=False,
                     int_cols=[],
                     int_props=[],
                     float_cols=[],
                     float_props=[],
                     str_cols=[],
                     str_props=[]):
"""
Args:
    object_type(string): 边类型或顶点类型
    expand_factor(int): 负采样个数
    strategy(string): 采样策略，支持random, in_degree, node_weight
    conditional(bool): 是否使用按条件负采样。按条件负采样时该值设为True
    unique(bool): 负样本是否需要是unique的。
    int_cols(list): 指定的int类型属性的下标，表示在这些指定的属性下进行负采样。比如输入的正样
        本对里dst_ids的int属性有3个，int_cols=[0,1]表示，在第一个int属性和dst_ids的第1个
        int属性一样的节点，以及第2个int属性和dst_ids的第2个属性一样的节点里选取负样本。
    int_props(list): int_cols里每个属性采样的比例。比如int_cols=[0,1],int_props=[0.1,0.2],
        表示在和dst_ids的第1个int属性一样的点里采样expand_factor*0.1个负样本，在和dst_ids的
        第2个int属性一样的点里采样expand_factor*0.2个负样本。
    float_cols(list): 指定的float类型属性的下标，同int_cols。
    float_props(list): float_cols的每个属性所占比例，同int_props。
    str_cols(list): 指定的string类型属性的下标，同int_cols。
    str_props(list): str_cols的每个属性所占比例，同int_props.
Return:
    NegativeSampler对象
"""
```

**注意：**<br />
负采样时，会在指定属性条件里按照strategy指定的策略负采样，要求sum(int_props) + sum(float_props) + sum(str_props) <= 1，如果该值<1，剩下的负样本采样时不再按照指定属性条件，只按照strategy采样。

- 接口<br />

```python
def get(src_ids, dst_ids):
""" 对指定的src_ids, dst_ids正样本对进行负采样。
Args:
    src_ids(numpy.ndarray): 一维int64数组，正样本源节点的ids
    dst_ids(numpy.ndarray): 一维int64数组，正样本目的节点的ids
Return:
    Nodes对象
"""
```

采样时会去除所有src_ids的全部邻居。<br />

- 示例<br />

```python
"""
假设点类型为item，它有3个int属性，1个float属性，1个string属性。
正样本为:
    src_ids = np.array([1,2,3,4,5])
    dst_ids = np.array([6,2,3,5,9])
现在需要在按照'node_weight'策略从给定的点表里进行负采样，并且要求在第1个int属性值等于dst_ids
的第1个int属性的点里采样2个负节点，在第1个string属性值等于dst_ids的第1个string属性值的点里采样
2个负样本
"""
s = g.negative_sampler('item',
                       expand_factor=4,
                       strategy='node_weight',
                       conditional=True，
                       unique=False,
                       int_cols=[0],
                       int_props=[0.5],
                       str_cols=[0],
                       str_props=[0.5])
src_ids = np.array([1,2,3,4,5])
dst_ids = np.array([6,2,3,5,9])
nodes = s.get(src_ids, dst_ids)
```
