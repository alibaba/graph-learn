# Quick Start

This document contains three sections

- How to quickly run through a GNN model based on **GL**

- How to load data into **GL** and how to use the graph data, graph sampling, negative sampling, and other interfaces

- Using **GraphSAGE** as an example, how to develop a GNN model of your own based on **GL** and TensorFlow


## Run through the built-in model

**GL** has some common models built-in, such as **GCN**, **GraphSAGE**, and datasets **cora, ppi**, etc.
We start our exposure to **GL** by running through the node classification task for **cora** data. For the complete model code, please refer to [examples](../../examples/tf).


``` shell
cd graph-learn/examples/data/
python cora.py
cd ../tf/ego_sage/
python train_supervised.py
```

## **GL** Interface Usage

**GL** provides a number of basic interfaces for GNN development, and we provide graph interface usage examples to show how to compose, query, sample, and negatively sample based on **GL**.

Before we start, we need to prepare a graph data source, here we prepare a script for generating data [gen_test_data.py](../../examples/basic/gen_test_data.py), which is used to generate local data for vertices and edges.

Prepare the test script as follows.
[test_dist_server_mode_fs_tracker.py](../../examples/basic/test_dist_server_mode_fs_tracker.py)
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

[query_examples.py](../../examples/basic/query_examples.py)More examples of the use of the graph interface are shown in the script for reference.

After preparing the data and code, we pull up 5 processes locally, 2 servers and 3 workers, for distributed execution.

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

## Develop a GNN model

The following will develop a supervised **GraphSAGE** model based on **GL** and **TensorFlow**, and train it on cora data.

### Data preparation

We use the open source dataset cora, which contains a number of papers on machine learning, as well as citation relationships between papers, each containing 1433 attributes. These papers can be classified into 7 categories: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory. this GNN task aims to predict the classification of papers. We processed the open source cora data to get the data format required for our composition. cora data download and processing script refer to [cora.py](../../examples/data/cora.py) .

```
cd graph-learn/examples/data
python cora.py
```

Produces edge data and vertex data. where edge data, i.e., citation relations between papers, where a paper is cited by at least one other paper.
The vertex data, i.e., the lexical representation of the paper, includes attributes and labels of the paper, with a total of 1433 dimensions for attributes and 7 categories for paper categories, so the label value domain is set to 0~6.

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


The vertex data contains labels and attributes in addition to ids, where attributes are 1433 floats. the edge data contains the weights of the edges in addition to the two endpoint ids.
The data format is described by the `gl.Decoder` class.

```python
import graphlearn as gl
node_decoder = gl.Decoder(labeled=True, attr_types=["float"] * args.features_num)
edge_decoder = gl.Decoder(weighted=True)
```

### Graph construction

Graph construction is the process of loading vertex data and edge data into memory and converting them into a logical graph format. After the build is complete, it is available for querying and sampling


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
g.init()
```

### Graph sampling
To implement GraphSAGE, graph sampling is required to serve as input to the upper layer network. Here, our sampling sequence is.
(1) Sampling the seed "item" vertices by batch.
(2) Sampling the 1-hop neighbors and 2-hop neighbors of the above vertices along the "relation" edge.
(3) Get the attributes of all vertices on the path and the labels of the seed vertices.
Here we define a graph sampling query to get the sample data of each iteration of batch by traversing the graph.

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

### Model code
- Define the loss and accuracy calculation functions, and define the train function to input the samples generated by query on the graph to the model.

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

- GNN model

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

- Start training

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
