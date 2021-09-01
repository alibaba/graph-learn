# 快速开始

本文档包含三个部分

- 如何基于**GL**快速跑通一个GNN模型

- 如何把数据加载到**GL**中，以及如何使用图数据、图采样、负采样等接口

- 以**GraphSAGE**为例，说明如何基于**GL**和TensorFlow开发一个自己的GNN模型


# 1. 跑通内置模型

**GL**内置了一些常见模型，如**GCN**，**GraphSAGE**，以及数据集**cora、ppi**等。
我们从跑通**core**数据的顶点分类任务开始接触**GL**。完整模型代码请参考[模型示例](model_examples.md)。<br />


``` shell
# 准备数据
cd graph-learn/examples/data/
python cora.py

# 训练、模型评估
cd ../tf/ego_sage/
python train_supervised.py
```

# 2. **GL**接口使用

**GL**为GNN的开发提供了大量基础接口，我们提供了图接口使用示例以展示如何基于**GL**来构图、查询、采样、负采样。

在开始前，我们需要准备一份图数据源，这里准备了一个生成数据的脚本[gen_test_data.py](../examples/basic/gen_test_data.py)，用于生成顶点和边的本地数据。

准备测试脚本[test_dist_server_mode_fs_tracker.py](../examples/basic/test_dist_server_mode_fs_tracker.py)如下：
``` python
import getopt
import os
import sys

import graphlearn as gl
from query_examples import *

def main(argv):
  cur_path = sys.path[0]

  server_count = -1
  client_count = -1
  tracker = ""
  job_name = ""
  task_index = -1

  opts, args = getopt.getopt(argv,
                             's:c:t:j:ti:',
                             ['server_count=', 'client_count=', 'tracker=',
                              'job_name=', 'task_index='])
  for opt, arg in opts:
    if opt in ('-s', '--server_count'):
      server_count = int(arg)
    elif opt in ('-c', '--client_count'):
      client_count = int(arg)
    elif opt in ('-t', '--tracker'):
      tracker = arg
    elif opt in ('-j', '--job_name'):
      job_name = arg
    elif opt in ('-ti', '--task_index'):
      task_index = int(arg)
    else:
      pass

  g = gl.Graph()

  g.node(os.path.join(cur_path, "data/user"),
         node_type="user", decoder=gl.Decoder(weighted=True)) \
    .node(os.path.join(cur_path, "data/item"),
          node_type="item", decoder=gl.Decoder(attr_types=['string', 'int', 'float', 'float', 'string'])) \
    .edge(os.path.join(cur_path, "data/u-i"),
          edge_type=("user", "item", "buy"), decoder=gl.Decoder(weighted=True), directed=False) 

  cluster={"server_count": server_count, "client_count": client_count, "tracker":tracker}
  g.init(cluster=cluster, job_name=job_name, task_index=task_index)

  if job_name == "server":
    print("Server {} started.".format(task_index))
    g.wait_for_close()

  if job_name == "client":
    print("Client {} started.".format(task_index))

    query = g.V("user").batch(32).shuffle(traverse=True).alias("src") \
          .outV("buy").sample(5).by("edge_weight").alias("hop1") \
          .inE("buy").sample(2).by("random").alias("hop1-hop2") \
          .inV().alias("hop2") \
          .values()

    ds = gl.Dataset(query)
    epoch = 2
    for i in range(epoch):
      step = 0
      while True:
        try:
          res = ds.next()
          src_nodes = res["src"]
          print(src_nodes.ids)
        except gl.OutOfRangeError:
          break

    g.close()
    print("Client {} stopped.".format(task_index))

if __name__ == "__main__":
  main(sys.argv[1:])
```

[query_examples.py](examples/basic/query_examples.py)脚本中展示了更多的图接口的使用示例以供参考。

准备完数据和代码后，我们在本地拉起5个进程，2个server，3个worker，分布式执行。

``` shell
#!/usr/bin/env bash
HERE=$(cd "$(dirname "$0")";pwd)

rm -rf $HERE/tracker
mkdir -p $HERE/tracker

# Only generating data when ./data folder is not existed.
# If `gen_test_data.py` is modified, then delete the data folder first.
if [ ! -d "$HERE/data" ]; then
  mkdir -p $HERE/data
  python $HERE/gen_test_data.py
fi

# Start a graphlearn cluster with 2 servers(processes) and 3 clients(processes).
python $HERE/test_dist_server_mode_fs_tracker.py \
  --server_count=2 --client_count=3 --tracker=$HERE/tracker --job_name="server" --task_index=0 &
sleep 1
python $HERE/test_dist_server_mode_fs_tracker.py \
  --server_count=2 --client_count=3 --tracker=$HERE/tracker --job_name="server" --task_index=1 &
sleep 1
python $HERE/test_dist_server_mode_fs_tracker.py \
  --server_count=2 --client_count=3 --tracker=$HERE/tracker --job_name="client" --task_index=0 &
sleep 1
python $HERE/test_dist_server_mode_fs_tracker.py \
  --server_count=2 --client_count=3 --tracker=$HERE/tracker --job_name="client" --task_index=1 &
sleep 1
python $HERE/test_dist_server_mode_fs_tracker.py \
  --server_count=2 --client_count=3 --tracker=$HERE/tracker --job_name="client" --task_index=2
```

# 3. 开发一个GNN模型

下面将基于**GL**和**TensorFlow**开发一个有监督的**GraphSAGE**模型，并在Cora数据上训练。<br />

## 3.1 数据准备

我们使用开源数据集Cora，它包含了机器学习的一些论文，以及论文之间的引用关系，每篇论文包含1433个属性。这些论文可以划分为7种类别：Case_Based，Genetic_Algorithms，Neural_Networks，Probabilistic_Methods，Reinforcement_Learning，Rule_Learning，Theory。该GNN任务的目的是预测论文的分类。我们将开源的Cora数据进行处理，得到我们构图所需的数据格式。Cora数据下载和处理的脚本参考[cora.py](../examples/data/cora.py)。

```
cd graph-learn/examples/data
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


顶点数据除了id以外，包含label和attributes，其中attributes为1433个float。边数据除了两个端点id以外，还包含边的权重。
数据格式通过`gl.Decoder`类描述。

```python
import graphlearn as gl

# 描述顶点表的数据格式，包含lable和attributes
node_decoder = gl.Decoder(labeled=True, attr_types=["float"] * args.features_num)

# 表示边表的数据格式，除了端点id以外，还有边的权重
edge_decoder = gl.Decoder(weighted=True)
```

## 3.2 图构建

图构建的过程是将顶点数据和边数据加载到内存中，转换为逻辑上的图格式。构建完成后，可供查询和采样。<br />


```python
import graphlearn as gl

def load_graph(args):
  dataset_folder = args.dataset_folder
  node_type = args.node_type
  edge_type = args.edge_type
  g = gl.Graph()                                                           \
        .node(dataset_folder + "node_table", node_type=node_type,
              decoder=node_decoder)                      \
        .edge(dataset_folder + "edge_table",
              edge_type=(node_type, node_type, edge_type),
              decoder=edge_decoder, directed=False)           \
        .node(dataset_folder + "train_table", node_type=node_type,
              decoder=gl.Decoder(weighted=True), mask=gl.Mask.TRAIN)       \
        .node(dataset_folder + "val_table", node_type=node_type,
              decoder=gl.Decoder(weighted=True), mask=gl.Mask.VAL)         \
        .node(dataset_folder + "test_table", node_type=node_type,
              decoder=gl.Decoder(weighted=True), mask=gl.Mask.TEST)
  return g


# 调用.init()进行初始化。这里以单机运行为例，分布式详见[图对象-初始化数据](graph_object_cn.md)。
g.init()
```

## 3.3 图采样
为了实现GraphSAGE，需要进行图采样以作为上层网络的输入。在这里，我们的采样顺序为：<br />
(1) 按batch采样种子“item”顶点；<br />
(2) 采样上述顶点沿着“relation”边的1-hop邻居和2-hop邻居；<br />
(3) 获取路径的上所有顶点的属性和种子顶点的labels。<br />

这里我们定义了一个图采样query，通过遍历图，得到每一次迭代的batch的样本数据。<br />

``` python
def query(graph, args):
  prefix = 'train'
  assert len(args.nbrs_num) == args.hops_num
  bs = args.train_batch_size
  q = graph.V(args.node_type, mask=gl.Mask.TRAIN).batch(bs).alias(prefix)
  for idx, hop in enumerate(args.nbrs_num):
    alias = prefix + '_hop' + str(idx)
    q = q.outV(args.edge_type).sample(hop).by('random').alias(alias)
  return q.values()
```

## 3.4 模型代码
- 定义loss和accuracy计算函数，并定义train函数，将图上query产生的样本输入给模型。

```python
def supervised_loss(logits, labels):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  return tf.reduce_mean(loss)

def accuracy(logits, labels):
  indices = tf.math.argmax(logits, 1, output_type=tf.int32)
  correct = tf.reduce_sum(tf.cast(tf.math.equal(indices, labels), tf.float32))
  return correct / tf.cast(tf.shape(labels)[0], tf.float32)

def train(graph, model, args):
  tfg.conf.training = True
  query_train = query(graph, args)
  dataset = tfg.Dataset(query_train, window=5)
  eg_train = dataset.get_egograph('train')
  train_embeddings = model.forward(eg_train)
  loss = supervised_loss(train_embeddings, eg_train.src.labels)
  return dataset.iterator, loss
```

- 定义GNN模型

```python
# ego_sage.py
import tensorflow as tf
import graphlearn.python.nn.tf as tfg


class EgoGraphSAGE(tfg.EgoGNN):
  def __init__(self,
               dims,
               agg_type="mean",
               bn_func=None,
               act_func=tf.nn.relu,
               dropout=0.0,
               **kwargs):
    assert len(dims) > 1

    layers = []
    for i in range(len(dims) - 1):
      conv = tfg.EgoSAGEConv("homo_" + str(i),
                             in_dim=dims[i],
                             out_dim=dims[i + 1],
                             agg_type=agg_type)
      # If the len(dims) = K, it means that (K-1) LEVEL layers will be added. At
      # each LEVEL, computation will be performed for each two adjacent hops,
      # such as (nodes, hop1), (hop1, hop2) ... . We have (K-1-i) such pairs at
      # LEVEL i. In a homogeneous graph, they will share model parameters.
      layer = tfg.EgoLayer([conv] * (len(dims) - 1 - i))
      layers.append(layer)

    super(EgoGraphSAGE, self).__init__(layers, bn_func, act_func, dropout)
```

- 开始训练

```python
def run(args):
  gl.set_tape_capacity(1)
  g = load_graph(args)
  g.init()
  # Define Model
  dims = [args.features_num] + [args.hidden_dim] * (args.hops_num - 1) \
        + [args.class_num]
  model = EgoGraphSAGE(dims,
                       agg_type=args.agg_type,
                       act_func=tf.nn.relu,
                       dropout=args.in_drop_rate)
  # train and test
  train_iterator, loss = train(g, model, args)
  optimizer=tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  train_op = optimizer.minimize(loss)
  train_ops = [loss, train_op]
  test_iterator, test_acc = test(g, model, args)
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.initializer)
    step = 0
    print("Start Training...")
    for i in range(args.epoch):
      try:
        while True:
          ret = sess.run(train_ops)
          print("Epoch {}, Iter {}, Loss {:.5f}".format(i, step, ret[0]))
          step += 1
      except tf.errors.OutOfRangeError:
        sess.run(train_iterator.initializer) # reinitialize dataset.
  g.close()


if __name__ == "__main__":
  cur_path = sys.path[0]
  argparser = argparse.ArgumentParser("Train EgoSAGE Supervised.")
  argparser.add_argument('--dataset_folder', type=str,
                         default=os.path.join(cur_path, '../../data/cora/'),
                         help="Dataset Folder, list files are node_table, edge_table, "
                              "train_table, val_table and test_table")
  argparser.add_argument('--class_num', type=int, default=7)
  argparser.add_argument('--features_num', type=int, default=1433)
  argparser.add_argument('--train_batch_size', type=int, default=140)
  argparser.add_argument('--hidden_dim', type=int, default=128)
  argparser.add_argument('--in_drop_rate', type=float, default=0.5)
  argparser.add_argument('--hops_num', type=int, default=2)
  argparser.add_argument('--nbrs_num', type=list, default=[25, 10])
  argparser.add_argument('--agg_type', type=str, default="gcn")
  argparser.add_argument('--learning_algo', type=str, default="adam")
  argparser.add_argument('--learning_rate', type=float, default=0.05)
  argparser.add_argument('--weight_decay', type=float, default=0.0005)
  argparser.add_argument('--epoch', type=int, default=40)
  argparser.add_argument('--node_type', type=str, default='item')
  argparser.add_argument('--edge_type', type=str, default='relation')
  args = argparser.parse_args()

  run(args)
```


