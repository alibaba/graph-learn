# 数据源

本文档用以说明GraphLearn支持的数据格式，以及如何通过API来描述和解析。<br />

# 1 数据格式
Graph数据可分为**顶点数据**和**边数据**。一般的，顶点数据包含**顶点ID**和**属性**，描述一个实体；边数据包含**源顶点ID**和**目的顶点ID**，描述顶点间的关系。在异构图场景中，顶点和边分别存在多种类型。因此，我们需要顶点和边的类型信息，才能对不同类型的顶点和边加以识别。类型信息通过API来描述。顶点和边都可以具有属性，比如“某用户在星期六上午购买了某商品”，时间信息“星期六上午”就是边属性。此外，很多场景用户需要“权重”的概念，或是顶点权重，或是边的权重，作为某种重要性的度量，比如“按权重进行邻居节点采样”。“权重”的来源多种多样，因任务不同而不同。在有监督学习的分类任务中，顶点或边还可能拥有标签。<br />
<br />我们将这些典型场景的数据格式抽象为**ATTRIBUTED**、**WEIGHTED、LABELED**，分别用于表示顶点或边包含属性的、具有权重的、具有标签的。对顶点数据源和边数据源来说，这三者可以同时存在，也可以部分存在。<br />

## 1.1 基础格式
基础的顶点数据只包含一个顶点的ID，ID类型为bigint，每条数据代表一个顶点。很多时候只有顶点ID是不够的，还需包含属性、权重或标签。<br />
<br />基础的边数据只包含源顶点ID和目的顶点ID，ID类型为bigint，每条数据代表一条边，表示两个顶点之间的关系。基础边数据源的schema如下所示。
基础的边数据格式可以独立使用，即不附加属性、权重和标签。<br />

边基础格式schema

| 域 | 数据类型 | 备注 |
| --- | --- | --- |
| src_id | BIGINT |  |
| dst_id | BIGINT |  |

<br />

## 1.2 属性格式（ATTRIBUTED）
用于表达顶点或边的属性信息。一般情况下，顶点默认具有属性，不然只需要边表就够了。属性列只有一列，为string类型。
string内部可通过自定义分隔符分割多个属性。比如，某一顶点属性有3个，分别为`shanghai, 10, 0.01`，用分隔符‘:’分隔，则该顶点对应的属性数据为`shanghai:10:0.01`。
当数据格式具有属性时，无论是顶点数据，还是边数据，在API描述时，都需要显示指定**ATTRIBUTED**以告知系统。

顶点数据属性格式schema

| 域 | 数据类型 | 备注 |
| --- | --- | --- |
| id | BIGINT |  |
| attributes | STRING |  |

<br />
边数据属性格式schema

| 域 | 数据类型 | 备注 |
| --- | --- | --- |
| src_id | BIGINT |  |
| dst_id | BIGINT |  |
| attributes | STRING |  |
<br />

## 1.3 权重格式（WEIGHTED）
用于表达顶点或边带有权重的情况。权重列只有一列，为**float**类型。当数据格式具有权重时，无论是顶点数据，还是边数据，在API描述时，都需要显示指定**WEIGHTED**以告知系统。

顶点数据权重格式schema

| 域 | 数据类型 | 信息列 |
| --- | --- | --- |
| id | BIGINT |  |
| attributes | FLOAT |  |

<br />
边数据权重格式schema

| 域 | 数据类型 | 备注 |
| --- | --- | --- |
| src_id | BIGINT |  |
| dst_id | BIGINT |  |
| weight | FLOAT  |  |
<br />

## 1.4 标签格式（LABELED）
用于表达顶点或边带有标签的情况。标签列只有一列，为int类型。当数据格式具有标签时，无论是顶点数据，还是边数据，在API描述时，都需要显示指定**LABELD**以告知系统。

顶点数据标签格式schema

| 域 | 数据类型 | 备注 |
| --- | --- | --- |
| id | BIGINT |  |
| label | INT |  |
<br />

边数据标签格式schema

| 域 | 数据类型 | 备注 |
| --- | --- | --- |
| src_id | BIGINT |  |
| dst_id | BIGINT |  |
| label | INT |  |
<br />

## 1.5 组合格式
ID是组成顶点和边数据源的必选信息，weight，label，attribute为可选信息。当同时具备**WEIGHTED、ATTRIBUTED、LABELED**一到多个时，在数据源中，必选信息和可选格式信息的组合需要遵循一定的顺序。<br />
<br />1）**顶点数据源**，混合格式schema的顺序如下表所示。<br />

| 域 | 数据类型 | 备注 |
| --- | --- | --- |
| id | BIGINT | 必选 |
| weight | FLOAT | 可选: WEIGHTED |
| label | BIGINT | 可选: LABELED |
| attributes | STRING | 可选: ATTRIBUTED |


<br />2）**边数据源**，混合格式schema的顺序如下表所示。<br />

| 域 | 数据类型 | 备注 |
| --- | --- | --- |
| src_id | BIGINT | 必选 |
| dst_id | BIGINT | 必选 |
| weight | FLOAT | 可选: WEIGHTED |
| label | BIGINT | 可选: LABELED |
| attributes | STRING | 可选: ATTRIBUTED |


<br />扩展信息可选择**0个或多个**，同时需要保证schema的**顺序维持上表顺序不变**。<br />

# 2 数据源类型

<div align=center> <img height ='320' src ="images/data_source.png" /> </div>

<br />系统抽象了数据接入层，可方便对接多种类型的数据源，目前支持LocalFileSystem，如果在阿里云PAI平台使用，可直读MaxCompute数据表。数据表现为二维结构化，行代表一个顶点或一条边数据，列表示顶点或边的某一项信息。<br />


<a name="9Im2p"></a>
## 2.1 Local FileSystem
在本地文件中，数据类型如下。其中，列名不做要求。支持从一个或多个本地文件读取数据，以便于在本地做调试。

| 列 | 类型 |
| --- | --- |
| id | int64 |
| weight | float |
| label | int32 |
| features | string |

<br />

- 顶点文件格式。其中，第一行为列名，表示必选信息或扩展信息，以**tab**分隔，每一个元素为“列名:数据类型”。其余每行数据代表一个顶点的信息，与第一列的信息名对应，以**tab**分隔。

```python
# file://node_table
id:int64  feature:string
0 shanghai:0:s2:10:0.1:0.5
1 beijing:1:s2:11:0.1:0.5
2 hangzhou:2:s2:12:0.1:0.5
3 shanghai:3:s2:13:0.1:0.5
```

<br />2）边文件格式。其中，第一行为列名，表示必选信息或扩展信息，以**tab**分隔，每一个元素为“列名:数据类型”。<br />其余每行数据代表一条边的信息，与第一列的信息名对应，以**tab**分隔。

```python
# file://edge_table
src_id:int64  dst_id:int64  weight:float  feature:string
0 5 0.215340  red:0:s2:10:0.1:0.5
0 7 0.933091  grey:0:s2:10:0.1:0.5
0 1 0.362519  blue:0:s2:10:0.1:0.5
0 9 0.097545  yellow:0:s2:10:0.1:0.5
```

<br />通过本地文件作为数据源，可以直接在脚本中使用文件路径。详见下一章“[图对象](graph_object_cn.md)”。<br />

<a name="mzVG6"></a>

## 2.2 阿里云MaxCompute数据表
在MaxCompute表中，数据类型如下。其中，列名不做要求。

| 列 | 类型 |
| --- | --- |
| id | BIGINT |
| weight | FLOAT |
| label | BIGINT |
| features | STRING |

使用MaxCompute作为数据源，需要以下两步：<br />1）通过PAI命令提交GL Job，将MaxCompute表作为`tables`参数的输入，多个表以逗号分隔。
```python
pai -name graphlearn
-Dscript=''
-DentryFile=''
-Dtables="odps://prj/tables/node_table,odps://prj/tables/edge_table"
...;
```

<br />2）在脚本中，通过TensorFlow的FLAG获取MaxCompute表参数，从而获取数据源，数据源可为一个或多个。
```python
import tensorflow as tf
import graphlearn as gl

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("tables", "", "odps table name")
node_source, edge_source = FLAGS.tables.split(',')
```
<a name="eh1Rf"></a>

# 3 用户API
<a name="IqJlv"></a>
## 3.1 Decoder定义
`Decoder`类用于描述上所述数据格式，定义如下。
```python
class Decoder(weighted=False, labeled=False, attr_types=None, attr_delimiter=":")
"""
weighted:       描述数据源是否带权重，默认为False
labeled:        描述数据源是否带有标签，默认为False
attr_types:     当数据源带属性时，该参数为一个string list，描述每一个属性的类型。
                list中的每个元素仅支持"string"、"int"和"float"类型。
                参数形如["string", "int", "float"]，代表数据的属性列包含有3个属性，
                按照顺序分别是string类型，int类型，float类型。
                默认None，即数据源不带属性。
attr_delimiter: 当数据带有属性(被压缩为一个大string)时，需要知道如何解析，该参数描述各个属性间的分隔符。
                如"shanghai:0:0.1"，分隔符为":"。默认为":"。
"""
```

<br />考虑与神经网络的结合，string类型的属性比较难处理，通常做法是先将string通过hash映射到int，再把int编码成embedding。为此，GL对string类型的属性做了特别扩展，即支持在图数据初始化阶段把string转成int。此时，attr_types参数中的"string"需要变为tuple类型`("string", bucket_size)`，bucket_size表示被转换到的int空间大小。若做此转换，后续访问时统一为int类型的attributes。除了简化后续操作以外，该转换也会大大降低内存开销。<br />

<a name="A2hT8"></a>
## 3.2 顶点Decoder
顶点的Decoder有以下几种形式。
```python
import graphlearn as gl

# schema = (src_id int64, dst_id int64, weight double)
gl.Decoder(weighted=True)

# schema = (src_id int64, dst_id int64, label int32)
gl.Decoder(labeled=True)

# schema = (src_id int64, dst_id int64, attributes string)
gl.Decoder(attr_type={your_attr_types}, attr_delimiter={you_delimiter})

# schema = (src_id int64, dst_id int64, weight float, attributes string)
ag.Decoder(weightd=True, attr_type={your_attr_types}, attr_delimiter={you_delimiter})

# schema = (src_id int64, dst_id int64, weight float, label int32)
gl.Decoder(weighted=True, labeled=True)

# schema = (src_id int64, dst_id int64, label int32, attributes string)
gl.Decoder(labeled=True, attr_type={your_attr_types}, attr_delimiter={you_delimiter})

# schema = (src_id int64, dst_id int64, weight float, label int32 attributes string)
gl.Decoder(weighted=True, labeled=True, attr_type={your_attr_types}, attr_delimiter={you_delimiter})
```


<a name="wKeTO"></a>
## 3.3 边Decoder
边的Decoder有以下几种形式。
```python
import graphlearn as gl

# schema = (scr_id int64, dst_id int64)
gl.Decoder()

# schema = (src_id int64, dst_id int64, weight float)
gl.Decoder(weighted=True)

# schema = (src_id int64, dst_id int64, label int32)
gl.Decoder(labeled=True)

# schema = (src_id int64, dst_id int64, attributes string)
gl.Decoder(attr_type={your_attr_types}, attr_delimiter={you_delimiter})

# schema = (src_id int64, dst_id int64, weight float, attributes string)
gl.Decoder(weightd=True, attr_type={your_attr_types}, attr_delimiter={you_delimiter})

# schema = (src_id int64, dst_id int64, weight float, label int32)
gl.Decoder(weighted=True, labeled=True)

# schema = (src_id int64, dst_id int64, weight float, label int32, attributes string)
gl.Decoder(weighted=True, labeled=True, attr_type={your_attr_types}, attr_delimiter={you_delimiter})

# schema = (src_id int64, dst_id int64, label int32, attributes string)
gl.Decoder(labeled=True, attr_type={your_attr_types}, attr_delimiter={you_delimiter})
```


<a name="BdNad"></a>
## 3.4 使用示例
假设数据源如下表1，表2，表3所示。<br />

表1 item顶点表

| id | feature |
| --- | --- |
| 10001 | feature1:1:0.1 |
| 10002 | feature2:2:0.2 |
| 10003 | feature3:3:0.3 |

表2 user顶点表

| id | feature |
| --- | --- |
| 123 | 0.1:0.2:0.3 |
| 124 | 0.4:0.5:0.6 |
| 125 | 0.7:0.8:0.9 |


表3 user-item边表

| src_id | dst_id | weight |
| --- | --- | --- |
| 123 | 10001 | 0.1 |
| 124 | 10001 | 0.2 |
| 124 | 10002 | 0.3 |

<br />对item顶点表构建`item_node_decoder`，对user顶点表构建`user_node_decoder`，边表构建`edge_decoder`，代码如下。

```python
import graphlearn as gl

item_node_decoder = gl.Decoder(attr_types=["string", "int", "float"])
user_node_decoder = gl.Decoder(attr_types=["float", "float", "float"])
edge_decoder = gl.Decoder(weighted=True)
```

<br />对每一个数据源构建完Decoder之后，在图中加入数据源，并指定对应的Decoder，详见[图对象](graph_object_cn.md) 。

