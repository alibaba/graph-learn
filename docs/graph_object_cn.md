# 图对象

Graph对象是将原始数据组织起来、供上层算子进行操作的本体。Graph对象支持同构图、异构图、属性图，图中的详细信息通过相关API来表达。一个Graph对象的构建大体包括**3步**：

- **声明Graph对象**
- **描述拓扑结构**
- **初始化数据**


<a name="2Kpiz"></a>
# 1. 声明Graph对象
声明Graph对象很简单，代码如下。后续所有相关操作都基于`g`来进行。
```python
import graphlearn as gl
g = gl.Graph()
```
<br />

# 2. 描述拓扑结构
拓扑结构描述的是图中的边与顶点的关联关系。这里的拓扑指的是“一类”数据的关系，而非“一条”数据。拓扑关系都是有向的，即有向图。<br />
<br />例如，对于一个“商品-商品”同构图，其拓扑结构图1所示。图中只有item到item类型的数据关联，边类型为swing，表示通过swing算法生成的item关联关系，源顶点和目的顶点类型均为item。

<div align=center> <img src ="images/i-i.png" /> <br /> 图1 item-item同构关系图<br /> </div>

<br />对于一个“用户-商品-商品”二部关系图，其拓扑结构如图2所示。图中包含两种类型的边，click边表示user与item的点击关系，源顶点类型为user，目的顶点类型为item；swing边表示item与item的关联关系，源顶点和目的顶点类型均为item。

<div align=center> <img src ="images/u-i-i.png" /> <br /> 图2 user-item-item异构关系图<br /> </div>

<br />这些点、边的类型将是进行**异构图**操作的依据，需要用户感知，并作为算子的输入。比如“采样某些user点击过的商品的关联商品”，那么系统会知道沿着“**user->click->item->swing->item**”去采样相关节点，而不是其他路径。<br />
<br />实际中，图中边的数量要远大于顶点的数量，大部分情况下顶点都具有丰富的属性信息，为了节省空间，边和顶点往往是分开存储的。我们通过向Graph对象添加顶点数据源和边数据源的形式，来构建拓扑结构。<br />

<a name="OVdVh"></a>
## 2.1 添加顶点
Graph对象提供 **node()** 接口，用于添加一种顶点数据源。**node()**返回Graph对象本身，也就意味着可以连续多次调用**node()**。具体接口形式和参数如下：
```python
def node(source, node_type, decoder)
''' 描述顶点类型与其数据schema的对应关系。

source:    string类型，顶点的数据源，详见“数据源”一章。
node_type: string类型，顶点类型；
decoder:   Decoder对象，用于描述数据源的schema；
'''
```

表1 带属性的顶点数据源

| id | attributes |
| --- | --- |
| 10001 | 0:0.1:0 |
| 10002 | 1:0.2:3 |
| 10003 | 3:0.3:4 |

表2 带权重的顶点数据源

| id | weight |
| --- | --- |
| 30001 | 0.1 |
| 30002 | 0.2 |
| 30003 | 0.3 |

如上表所示的数据源，可以通过如下代码添加到Graph对象中。当存在多种类型的顶点时，请注意每次调用`node()`时的node_type不能相同。
```python
g.node(source="table_1", node_type="user", decoder=Decoder(attr_types=["int", "float", "int"])) \
 .node(source="table_2", node_type="movie", decoder=Decoder(weighted=True)
```


<a name="grjJs"></a>
## 2.2 添加边
Graph对象提供 **edge()** 接口，用于添加一种边数据源，支持将同构或异构的边指定为无向边。**edge()**返回Graph对象本身，也就意味着可以连续多次调用**edge()**。通过添加边数据源，确定了图中边类型与其源点、目的点类型的对应关系，再结合对应的顶点类型数据源，共同构成一张打通连接关系的大图。具体接口形式和参数如下：
```python
def edge(source, edge_type, decoder, directed=True)
''' 描述边类型和其源顶点、目的顶点类型的对应关系，以及边类型与数据schema的对应关系。

source:    string类型，边的数据源，详见“数据源”一节。
edge_type: tuple，内容为(源点类型, 目的点类型, 边类型)3元组。
decoder:   Decoder对象，用于描述数据源的schema；
directed:  boolean, 边是否为无向边。默认True，为有向边。当为无向边时，采样必须通过GSL接口；
'''
```

表3 带权重的边数据源

| src_id | dst_id | weight |
| --- | --- | --- |
| 10001 | 10002 | 0.1 |
| 10002 | 10001 | 0.2 |
| 10003 | 10002 | 0.3 |
| 10004 | 10003 | 0.4 |

表4 带属性的边数据源

| src_id | dst_id | weight | attributes |
| --- | --- | --- | --- |
| 20001 | 30001 | 0.1 | 0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19 |
| 20001 | 30003 | 0.2 | 0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29 |
| 20003 | 30001 | 0.3 | 0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39 |
| 20004 | 30002 | 0.4 | 0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49 |

如上表所示的边数据源，可以通过如下代码添加到Graph对象中。当存在多种类型的边时，请注意每次调用`edge()`时的edge_type不能相同。

```python
ui_decoder = Decoder(weighted=True)
uv_decoder = Decoder(weighted=True, attr_types=["float"] * 10, attr_delimiter=',')

g.edge(source="table_3", edge_type=("user", "item", "click"), decoder=ui_decoder)
 .edge(source="table_4", edge_type=("user", "movie", "click_v"), decoder=uv_decoder)
```


<a name="HNiIP"></a>
# 3. 初始化

顶点与边添加完成后，需要调用初始化接口，完成从原始数据到内存索引的构建。初始化过程决定了图数据被Serving的情况，单机的还是分布式的。若为分布式的，还要区分Server Mode和Client-Server Mode。初始化完成后，便可对Graph对象进行操作了。<br />

<a name="nGHkF"></a>

## 3.1 单机
单机模式比较简单，表示该Graph对象Hold全部图数据。

```python
g.init()
```

<a name="oKpvB"></a>
## 3.2 分布式--Server Mode

该模式下，数据分布式存在于各个Server上，Server之间两两互联，每个Server对应一个Graph对象的入口。当进行图采样或查询等操作时，Graph对象把请求提交给本地Server，由Server决定如何分布式处理。Graph对象与本地Server之间不存在网络通信。

```python
g.init(task_index, task_count)
```
Server Mode适用于图规模不是超大的情况，此时，与分布式训练结合，例如TensorFlow，每个GL的Server需要与TensorFlow的worker在同一个进程，这就省去了worker到GL的server的一次数据通信开销。由于分布式规模不是超大（worker数不是很大），Server之间的两两互联不会造成网络的负担。另外，对于模型并行训练的场景，非典型的worker-ps模式也需要使用Server Mode部署。

与TensorFlow结合，大致代码如下。

```python
if FLAGS.job_name == "worker":
  g.init(task_index=FLAGS.task_index, task_count=len(FLAGS.worker_hosts.split(','))
  # Your model, use g to do some operation, such as sampling
  g.close()
else:
  # ps.join()
```
<br />
<a name="xgEg9"></a>

## 3.3 分布式--Client/Server Mode

该模式下，与Server Mode类似，数据分布式存在于各个Server上，Server之间两两互联。此时，Graph对象的入口位于Client端，而非Server端。每个Client都与唯一一个Server连接，该Server作为Client的响应Server（类似于Server Mode里的本地Server）。Client与Server的对应关系由负载均衡算法决定。Client作为入口，提交请求到其响应Server，由该Server决定如何分布式处理。<br />
<br />C/S Mode适用于分布式规模超大的情况（一百GB以上），此时以为worker规模很大，使用Server Mode部署会大大增加网络互联的开销。另外，图数据的规模和worker规模不一定匹配，例如，当1000个worker并发训练时，并不一定需要这么多Server去承载Graph数据，数据太分散会严重降低性能。一般而言，训练的worker数 >= 图Server数。<br />
<br />C/S Mode部署代码如下。

```python
g.init(cluster, job_name, task_index)
"""
cluster(dict): client与server的数量，例如cluster={"server_count": 2, "client_count": 4}
job_name(string): 角色类型，取值为"client"或"server"
task_index(int): 当前角色中的第几个
"""
```

<br />与TensorFlow结合，Client位于worker端，Server位于ps端，或将Server单独放置到TensorFlow的其他role里。大致代码如下。
```python
cluster={"server_count": 2, "client_count": 4}

if FLAGS.job_name == "worker":
    g.init(cluster=cluster, job_name="client", task_index=FLAGS.task_index)
    # Your model, use g to do some operation, such as sampling
    g.close()
else if FLAGS.job_name == "ps":
  g.init(cluster=cluster, job_name="server", task_index=FLAGS.task_index)
    g.wait_for_close()
    # ps.join()
else:
    # others
```

<br />**请注意，无论单机还是分布式，在使用完毕时需要显示调用`g.close()`。**

