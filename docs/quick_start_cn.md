# 快速开始

本文档包含三个部分

- 如何基于**GL**快速跑通一个GNN模型

- 如何把数据加载到**GL**中，以及如何使用图数据、图采样、负采样等接口

- 以**GraphSAGE**为例，说明如何基于**GL**和TensorFlow开发一个自己的GNN模型


# 1 跑通内置模型

**GL**内置了一些常见模型，如**GCN**，**GraphSAGE**，以及数据集**cora、ppi**等。
我们从跑通**core**数据的顶点分类任务开始接触**GL**，完整模型代码请参考[模型示例](model_examples.md)。<br />


``` shell
# 准备数据
cd graph-learn/examples/data/
python cora.py

# 训练、模型评估
python train_supervised.py
```


# 2 **GL**接口使用

**GL**为GNN的开发提供了大量基础接口，下面将会展示如何基于**GL**来构图、查询、采样、负采样。

我们准备了一个生成数据的脚本[gen_test_data.py](../examples/basic/distribute/gen_test_data.py)，用于生成顶点和边的本地数据。
在下面的分布式代码中，各个进程都读取同一个数据文件，确保所有进程都有权限访问该文件。测试代码如下。

``` python
# test.py
import sys, os
import getopt

import graphlearn as gl
import numpy as np

def main(argv):
  cur_path = sys.path[0]

  cluster = ""
  job_name = ""
  task_index = 0

  opts, args = getopt.getopt(argv, 'c:j:t:', ['cluster=', 'job_name=','task_index='])
  for opt, arg in opts:
    if opt in ('-c', '--cluster'):
      cluster = arg
    elif opt in ('-j', '--job_name'):
      job_name = arg
    elif opt in ('-t', '--task_index'):
      task_index = int(arg)
    else:
      pass

  # init distributed graph
  g = gl.Graph() \
       .node(os.path.join(cur_path, "data/user"),
            node_type="user", decoder=gl.Decoder(weighted=True)) \
       .node(os.path.join(cur_path, "data/item"),
             node_type="item",
             decoder=gl.Decoder(attr_types=['string', 'int', 'float', 'float', 'string'])) \
       .edge(os.path.join(cur_path, "data/u-i"),
             edge_type=("user", "item", "buy"), decoder=gl.Decoder(weighted=True), directed=False)
  g.init(cluster=cluster, job_name=job_name, task_index=task_index)
  # g.init() # For local.

  if job_name == "server":
    g.wait_for_close()

  if job_name == "client":
    # Lookup a batch of user nodes with given ids.
    nodes = g.V("item", feed=np.array([100, 102])).emit()
    print(nodes.ids)
    print(nodes.int_attrs)

    # Iterate users and random sample items the user has buy.
    q = g.V("user").batch(4) \
             .outV("buy").sample(2).by("random") \
             .values(lambda x: (x[0].weights, x[1].ids))
    while True:
      try:
        print(g.run(q))
      except gl.OutOfRangeError:
        break

    # Random sample seed buy edge from graph,
    # and sample the users who did not buy the items from the seed edges. 
    q = g.E("buy").batch(4).shuffle() \
         .inV() \
         .inNeg("buy").sample(3).by("in_degree") \
         .values(lambda x: x[2].ids)
    print(g.run(q))

    g.close()
if __name__ == "__main__":
  main(sys.argv[1:])
```


准备完数据和代码后，我们在本地拉起4个进程，2个worker2个server，分布式执行。

``` shell
HERE=$(cd "$(dirname "$0")";pwd)

rm -rf $HERE/.tmp/tracker
mkdir -p $HERE/.tmp/tracker

if [ ! -d "$HERE/data" ]; then
  mkdir -p $HERE/data
  python $HERE/gen_test_data.py # 你需要把gen_test_data.py、test.py放到shell脚本同一目录下
fi

python $HERE/test.py \
  --cluster="{\"client_count\": 2, \"server_count\": 2, \"tracker\": \"$HERE/.tmp/tracker\"}" \
  --job_name="server" --task_index=0 &
sleep 1
python $HERE/test.py \
  --cluster="{\"client_count\": 2, \"server_count\": 2, \"tracker\": \"$HERE/.tmp/tracker\"}" \
  --job_name="server" --task_index=1 &
sleep 1
python $HERE/test.py \
  --cluster="{\"client_count\": 2, \"server_count\": 2, \"tracker\": \"$HERE/.tmp/tracker\"}" \
  --job_name="client" --task_index=0 &
sleep 1
python $HERE/test.py \
  --cluster="{\"client_count\": 2, \"server_count\": 2, \"tracker\": \"$HERE/.tmp/tracker\"}" \
  --job_name="client" --task_index=1
```

# 3 开发一个GNN模型

下面将基于**GL**和**TensorFlow**开发一个有监督的**GraphSGAE**模型，并在Cora数据上训练，更详细的参考[模型的开发](algo_cn.md)。<br />

## 3.1 数据准备

我们使用开源数据集Cora，它包含了机器学习的一些论文，以及论文之间的引用关系，每篇论文包含1433个属性。这些论文可以划分为7种类别：Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory。该GNN任务的目的是预测论文的分类。我们将开源的Cora数据进行处理，得到我们构图所需的数据格式。Cora数据下载和处理的脚本参考[cora.py](../examples/data/cora.py)。
```
cd ../examples/data
python cora.py
```

产出边数据和顶点数据。其中，边数据即论文之间的引用关系，一篇论文由其他至少一篇论文引用；
顶点数据，即论文的词汇表示，包括论文的属性和标签，属性总共1433个维度，论文类别有7类，因此label值域设置为0~6。

```shell
src_id:int64   dst_id:int64
35  1033
35  103482
35  103515
```

```
id:int64  label:int32   feature:string
31336      4    0.0:0.0:... 
1061127    1    0.0:0.05882353:...
1106406    2    0.0:0.0:...
```


顶点顶点数据除了id以外，包含label和attributes，其中attributes为1433个float。边数据除了两个端点id以外，还包含边的权重。
数据格式通过`gl.Decoder`类描述。

```python
import graphlearn as gl

N_FEATURE = 1433

# 描述顶点表的数据格式，包含lable和attributes
node_decoder = gl.Decoder(labeled=True, attr_types=["float"] * N_FEATURE)

# 表示边表的数据格式，除了端点id以外，还有边的权重
edge_decoder = gl.Decoder(weighted=True)
```

## 3.2 图初始化

图初始化的过程是将顶点数据和边数据加载到内存中，转换为逻辑上的图格式。初始化完成后，可供查询和采样。<br />


```python
import graphlearn as gl

# 配置参数
N_CLASS = 7
N_FEATURE = 1433
BATCH_SIZE = 140
HIDDEN_DIM = 128
N_HOP =  2
HOPS = [10, 5]
N_EPOCHES = 2
DEPTH = 2

# 定义一个Graph对象 
g = gl.Graph()

# 通过`.node()`将顶点表加入到图中，并指定顶点的类型；这里只有一种类型的顶点，我们命名为"item"。
# 通过`.edge()`将边表加入到图中，并通过一个三元组描述类型，分别为源顶点类型、目的顶点类型和边类型。
g.node("examples/data/cora/node_table",
       node_type="item",
       decoder=gl.Decoder(labeled=True, attr_types=["float"] * N_FEATURE)) \
  .edge("examples/data/cora/edge_table",
        edge_type=("item", "item", "relation"), decoder=gl.Decoder(weighted=True), directed=False) \


# 调用.init()进行初始化。这里以单机运行为例，分布式详见[图对象-初始化数据](graph_object_cn.md)。
g.init()
```

## 3.3 图采样
为了实现GraphSAGE，需要进行图采样以作为上层网络的输入。在这里，我们的采样顺序为：<br />
(1) 按batch采样种子“item”顶点；<br />
(2) 采样上述顶点沿着“relation”边的1-hop邻居和2-hop邻居；<br />
(3) 获取路径的上所有顶点的属性和种子顶点的labels。<br />

这里我们定义了一个生成器，通过遍历图，得到每一次迭代的batch的样本数据。<br />

``` python
def sample_gen():
  query = g.V('item').batch(BATCH_SIZE) \
               .outV("relation").sample(10).by("random") \
               .outV("relation").sample(5).by("random") \
               .values(lambda x: (x[0].float_attrs, x[1].float_attrs, x[2].float_attrs, x[0].labels))
  while True:
    try:
      res = g.run(query)
      if res[0].shape[0] < BATCH_SIZE:
        break
      yield tuple([res[0].reshape(-1, N_FEATURE)]) + tuple([res[1].reshape(-1, N_FEATURE)]) \
            + tuple([res[2].reshape(-1, N_FEATURE)]) + tuple([res[3]])
    except gl.OutOfRangeError:
      break
```

## 3.4 模型代码
以TensorFlow Estimator为例，说明在**GL**上自行开发GNN的方式。<br />

- 将图采样的样本生成器作为`input_fn`

```python
import tensorflow as tf
def sample_input_fn():
  ds = tf.data.Dataset.from_generator(
    sample_gen,
    tuple([tf.float64] * 3) + tuple([tf.int32]),
    tuple([tf.TensorShape([BATCH_SIZE, N_FEATURE])]) + \
    tuple([tf.TensorShape([BATCH_SIZE *  HOPS[0], N_FEATURE])] ) + \
    tuple([tf.TensorShape([BATCH_SIZE * HOPS[0] * HOPS[1], N_FEATURE])]) + \
    tuple([tf.TensorShape([BATCH_SIZE])])
  )
  value = ds.repeat(N_EPOCHES).make_one_shot_iterator().get_next()
  layer_features = value[:3]
  features, labels = encode_fn(layer_features, 0, DEPTH), value[3]
  return {"logits": features}, labels
```

- 定义GNN模型的Aggregator和Encoder。

```python
vars = {}
def aggregate_fn(self_vecs, neigh_vecs, raw_feat_layer_index, layer_index):
  with tf.variable_scope(str(layer_index) + '_layer', reuse=tf.AUTO_REUSE):
    vars['neigh_weights'] = tf.get_variable(shape=[N_CLASS, N_CLASS], name='neigh_weights')
    vars['self_weights'] = tf.get_variable(shape=[N_CLASS, N_CLASS], name='self_weights')
    output_shape = self_vecs.get_shape()
    dropout = 0.5
    neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - dropout)
    self_vecs = tf.nn.dropout(self_vecs, 1 - dropout)
    neigh_vecs = tf.reshape(neigh_vecs,
                            [-1, HOPS[raw_feat_layer_index], N_CLASS])
    neigh_means = tf.reduce_mean(neigh_vecs, axis=-2)

    from_neighs = tf.matmul(neigh_means, vars['neigh_weights'])
    from_self = tf.matmul(self_vecs, vars["self_weights"])

    output = tf.add_n([from_self, from_neighs])
    output = tf.reshape(output, shape=[-1, output_shape[-1]])
    return tf.nn.leaky_relu(output)


def encode_fn(layer_features, raw_feat_layer_index, depth_to_encode):
  if depth_to_encode > 0:
    h_self_vec = encode_fn(layer_features, raw_feat_layer_index, depth_to_encode - 1)
    h_neighbor_vecs = encode_fn(layer_features, raw_feat_layer_index + 1, depth_to_encode - 1)
    return aggregate_fn(h_self_vec, h_neighbor_vecs, raw_feat_layer_index, depth_to_encode)
  else:
    h_self_vec = tf.cast(layer_features[raw_feat_layer_index], tf.float32)
    h_self_vec = tf.layers.dense(h_self_vec, N_CLASS, activation=tf.nn.leaky_relu)
  return h_self_vec
```

- 定义features_column

```python
features, labels = sample_input_fn()
feature_columns = []
for key in features.keys():
  feature_columns.append(tf.feature_column.numeric_column(key=key))
```

- 定义Loss和Model
```python
def loss_fn(logits, labels):    
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

def model_fn(features, labels, mode, params):
    logits = features['logits']
    loss = loss_fn(logits, labels)
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    spec = tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=train_op)
    return spec
```
- 实例化Estimator，并训练

```python
params = {"learning_rate": 1e-4,
          'feature_columns': feature_columns}

model = tf.estimator.Estimator(model_fn=model_fn,
                               params=params)
model.train(input_fn=sample_input_fn)
```


