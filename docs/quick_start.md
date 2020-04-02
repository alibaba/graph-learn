# Quick Start

In this tutorial, we will show you how to build a graph learning model
based on the low-level **GL** APIs and deep learning framework backend
like TensorFlow. We take GCN, one of the most popular graph neural network models, 
as an example to show details.

In the [Model Programming](model_programming.md) tutorial, we have 
introduced some fundamental concepts of **GL**, such as `EgoGraph` and `EgoTensor`.
We highly recommend you to go through it before continuing.

## How to build a learning based model

Generally, you need to implement the following four steps

- Sampling: Sample sub-graphs (which is named `EgoGraph`) using build-in sample functions. 

    We have abstracted four basic functions including `sample_seed`, `positive_sample`,
    `negative_sample` and `receptive_fn`. `sample_seed` used to generate seed `Nodes` or
    `Edges`(a batch of nodes or edges), and then `positve_sample` take
    them as input to generate positive sample `Edges`. The `negative_sample` function samples
    negative `Nodes` or `Edges` for unsupervised models. GNNs need to aggregate
    neighbors' information of nodes(edges) to update nodes(edges) embeddings, so we provide  `receptive_fn` to sample neighbors. The seed `Nodes`/`Edges` and sampled neighbors are organized as
    `EgoGraph`.


- Graph flow: Use `EgoFlow` to convert `EgoGraph` to `EgoTensor` according to different backends.

   **GL** models ard built on top of deep learning framework such as TensorFlow. So sampled
   EgoGraphs needs to be converted into tensor format `EgoTensor`. We wrapped `EgoFlow` for this
   conversion. `EgoFlow` also generates an iterator used for batch training and pipeline.
  
- Define encoders: Use `EgoGraph` encoders and feature encoders to encode `EgoTensor`.

    After getting `EgoTensor`,
    we need to define the transformation routine from raw data to embeddings. For GNN
    models, this step is to aggregate neighbors and combing it with self nodes/edges.
    
- Define loss function and training: Feed encoded embeddings to loss function and training.

    **GL** has built-in several common loss functions and optimizers, and you can also
    customize yours. Local and distributed training is supported as well.

We will detail these four steps and show you how to implement a GCN model.

### Sampling

We use the Cora dataset as an example, and we have provided a transform script `cora.py` 
to transform text files to **GL** specific format. After running this script, 
you will get five files: 
node_table, edge_table_with_self_loop, train_table, val_table and test_table, respectively. 
The first two files record the nodes and edges, and the last three files indicate nodes' roles.

The following code will load this graph into the machine memory:

```python
g = gl.Graph()\
      .node(dataset_folder + "node_table", node_type=node_type,
            decoder=gl.Decoder(labeled=True,
                               attr_types=["float"] * (config['features_num']),
                               attr_delimiter=":"))\
      .edge(dataset_folder + "edge_table_with_self_loop", 
            edge_type=(node_type, node_type, edge_type),
            decoder=gl.Decoder(weighted=True), directed=False)\
      .node(dataset_folder + "train_table", node_type="train",
            decoder=gl.Decoder(weighted=True))\
      .node(dataset_folder + "val_table", node_type="val",
            decoder=gl.Decoder(weighted=True))\
      .node(dataset_folder + "test_table", node_type="test",
            decoder=gl.Decoder(weighted=True))
```

Returned `g` is a **GL** `Graph` object.

GCN inherits from `LearningBasedModel` class, which encapsulates many common routines and 
facilitate the programming process.

```py
import graphlearn as gl
class GCN(gl.LearningBasedModel):
  def __init__(self,
               graph,
               output_dim,
               features_num,
               batch_size,
               categorical_attrs_desc='',
               hidden_dim=16,
               hops_num=2,):
  self.graph = graph
  self.batch_size = batch_size
```

We need to choose seed nodes and then provide the sampler strategy to form `EgoGraph` for 
these seed nodes. All of these are member functions of `LearningBaseModel` 
classes to be overwritten.

```python
class GCN(gl.LearningBasedModel):
  # ...
  def _sample_seed(self):
      return self.graph.V('train').batch(self.batch_size).values()

  def _positive_sample(self, t):
      return gl.Edges(t.ids, self.node_type,
                      t.ids, self.node_type,
                      self.edge_type, graph=self.graph)

  def _receptive_fn(self, nodes):
      return self.graph.V(nodes.type, feed=nodes).alias('v') \
        .outV(self.edge_type).sample().by('full').alias('v1') \
        .outV(self.edge_type).sample().by('full').alias('v2') \
        .emit(lambda x: gl.EgoGraph(x['v'], [ag.Layer(nodes=x['v1']), ag.Layer(nodes=x['v2'])]))
```

The first two functions are used to provide seed nodes for training.
`_receptive_fn` is used to form `EgoGraph`, and its parameter `nodes` are a batch 
of seed nodes returned from `_sample_seed` function.
 `outV` returns seed nodes' one-hop neighbor. 
 We can perform sampling using different sampling strategies, 
 here we use 'full' to return all 1-hop nodes. 
 `outV` is repeated again to get 2-hop neighbors of seed nodes.
  Lastly, seed nodes, as well as its neighbor nodes, are formed as `EgoGraph`.

 

### Graph flow

The `EgoGraph` are returned in NumPy format. 
To use it in a deep learning framework backend such as TensorFlow, 
we provide a transformation routine to convert these data into 
the corresponding tensor format. We call this transformation routine 
`EgoFlow` and it is defined in the `build` function:

```python
class GCN(gl.LearningBasedModel):
  def build(self):
    ego_flow = gl.EgoFlow(self._sample_seed,
                          self._positive_sample,
                          self._receptive_fn,
                          self.src_ego_spec)
    iterator = ego_flow.iterator
    pos_src_ego_tensor = ego_flow.pos_src_ego_tensor
    # ...
```

You can get `EgoTensor`s from `EgoFlow` corresponding to the former `EgoGraph`s.

 

### Define encoders.

The next step is to define the encoders of `EgoTensor` 
to encode Tensors to embeddings. 
Basically, there are two kinds of encoders, 
feature encoder is used to transform raw input features to numerical embeddings, 
and graph encoders is used to aggregate neighbor nodes' embeddings to seed nodes (egos). 
All the encoders are implemented in the member function `_encoders`:

```python
class GCN(gl.LearningBasedModel):
  def _encoders(self):
    depth = self.hops_num
    feature_encoders = [gl.encoders.IdentityEncoder()] * (depth + 1)
    conv_layers = []
    # for input layer
    conv_layers.append(gl.layers.GCNConv(self.hidden_dim))
    # for hidden layer
    for i in range(1, depth - 1):
      conv_layers.append(gl.layers.GCNConv(self.hidden_dim))
    # for output layer
    conv_layers.append(gl.layers.GCNConv(self.output_dim, act=None))
    encoder = gl.encoders.SparseEgoGraphEncoder(feature_encoders,
                                                  conv_layers)
    return {"src": encoder, "edge": None, "dst": None}
```

For Cora dataset, we do not need to do any transformation for raw feature data, 
because they are already numerical. 
Then we define the graph encoder by stacking convolutional layers. 
We predefined a lot of convolutional layers, including `GCNConv`, `GATConv`, etc. 

### Define loss function and training

Given defined encoders, now let's back to the `build` function to compute the loss.

```python
class GCN(gl.LearningBasedModel):
  # ...
  def _supervised_loss(self, emb, label):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(emb, label))

  def build(self):
    ego_flow = gl.EgoFlow(self._sample_seed,
                          self._positive_sample,
                          self._receptive_fn,
                          self.src_ego_spec,
                          full_graph_mode=self.full_graph_mode)
    iterator = ego_flow.iterator
    pos_src_ego_tensor = ego_flow.pos_src_ego_tensor
    src_emb = self.encoders['src'].encode(pos_src_ego_tensor)
    labels = pos_src_ego_tensor.src.labels
    loss = self._supervised_loss(src_emb, labels)

    return loss, iterator
```

The `loss` and `iterator` will be used in the following training process. We wrapped
some TensorFlow loss functions and optimizers for convenience. Local and distributed 
training is also provided by trainers like `LocalTFTrainer`.

```python
# gcn/train_supervised.py
from gcn import GCN
def train(config, graph)
  def model_fn():
    return GCN(graph, ...)
  trainer = gl.LocalTFTrainer(model_fn, epoch=200)
  trainer.train()

def main():
  config = {...}
  g = load_graph(config)
  g.init(server_id=0, server_count=1, tracker='../../data/')
  train(config, g)
```

Here we have provided a general idea on how to build a GCN model from scratch. 
For the full code, please check the examples/GCN folder.

We also implement common model examples including GCN, 
GAT, GraphSage, DeepWalk, LINE, TransE, Bipartite GraphSage, sample-based GCN and GAT.
 The best place to start is with these examples.

[Home](../README.md)
