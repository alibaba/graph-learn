# 负采样

<a name="znmkl"></a>
# 1 介绍
负采样作为无监督训练的重要手段，是指采样与给定顶点没有直接边关系的顶点。同邻居采样类似，负采样也有不同的实现策略，如随机、按顶点入度等。作为GNN的一种常见算子，负采样支持扩展和面向场景的自定义。<br />

<a name="6WYEX"></a>
# 2 用法
<a name="keU3J"></a>
## 2.1 接口
负采样算子以边或顶点类型作为输入，用于表达“在某种边类型下采样与给定顶点无直接关联的顶点”，当为顶点类型时，需要用户指定候选顶点集，框架不再考虑边关系。采样结果组织成`Nodes`对象（类似一跳邻居采样，但不存在`Edges`对象）。一个负采样操作可具体分为以下3步实现：

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

<br />在GSL中，实现负采样主要为 `outNeg()` / `inNeg()` / `Neg()`几个操作。
```python
# 1. 负采样一跳邻居顶点
g.V().outNeg(edge_type).sample(count).by(strategy)

# 2. 负采样二跳邻居顶点
g.V().outNeg(edge_type).sample(count).by(strategy).outNeg(edge_type).sample(count).by(strategy)

# 3. 在给定顶点候选集上负采样
g.V().Neg(node_type).sample(count).by("node_weight")
```


<a name="ePTLM"></a>
# 3 负采样策略
GL目前已支持以下几种负采样策略，对应产生`NegativeSampler`对象时的`strategy`参数。

| **strategy** | **说明** |
| --- | --- |
| random | 随机负采样，不保证true-negative |
| in_degree | 以顶点入度分布为概率进行负采样，保证true-negative |
| node_weight | 以顶点权重为概率进行负采样样 |

