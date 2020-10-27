# Userdefined Algorithm

In this document, we will introduce how to use the basic APIs provided by **GL** to cooperate with deep learning engines (such as TensorFlow) to build graph learning algorithms.
We demostrate the GCN model as an example which is the most popular model in graph nerual network. 
In [Algorithm Programming Paradigm](model_programming_cn.md), we introduced some basic concepts used in developing algorithms, such as ʻEgoGraph`, ʻEgoTensor` encoder, etc.,
Please understand these basic concepts before continuing reading.

# How to Build a Graph Learning Algorithm

In general, it requires following four steps to build an algorithm

- Specify sampling mode: use graph sampling and query methods to sample subgraphs and organize them into `EgoGraph`

    We abstracted 4 basic functions, `sample_seed`, `positive_sample`,
    `negative_sample` and `receptive_fn`.
    To generate `Nodes` ot `Edges`, we use `sample_seed` to tranverse the graph. Then, we use `positive_sample` with `Node` or `Edges` as inputs to generate the positive sample for tranining. For unsupervised learning `negative_sample` produces negative samples.
    GNNs need to aggregate neighbor information so that we abstract `receptive_fn` to sample neighbors. Finally, these seeds `Nodes`, `Edges` and sampled neighbors form an `EgoGraph`.

- Construct graph data flow: convert `EgoGraph` to `EgoTensor` using `EgoFlow`

    **GL** algorithm model is based on a deep learning engine similar to TensorFlow. 
    As a result, it requires to convert the sampled `EgoGraph`s to the tensor format `EgoTensor`, which is encapsulated in `EgoFlow`.
    `EgoFlow` can generate an iterator for batch training.

- Define encoder: Use `EgoGraph` encoder and feature encoder to encode `EgoTensor`.

    After getting the `EgoTensor`, we first encode the original point and edge features into vectors with some common feature encoders.
    These vectors can be used as the feature input of GNNs model. Then, we use the graph encoder to process the `EgoTensor`, combining the neighbor node features with its own characteristics to get the final nodes or edge vector.

- Design loss function and training process: select the appropriate loss function and write the training process.

    **GL** has built-in some common loss functions and optimizers, and encapsulates the training process. It supports both stand-alone and distributed training.
    Users can also customize the loss function, optimizer and training process.

Next, we introduce how to implement a GCN model using the above 4 steps.


### Sampling

We use the Cora dataset as the node classification example. We provide a simple data conversion script `cora.py` to convert the original Cora to the format required by **GL**. The script generates following 5 files: node_table, edge_table_with_self_loop, train_table, val_table and test_table.
They are the node table, the edge table, and the nodes tables used to distinguish training, validation, and testing sets.

Then, we can construct the graph using the following code.

```python
g = gl.Graph()\
      .node(dataset_folder + "node_table", node_type=node_type,
            decoder=gl.Decoder(labeled=True,
                               attr_types=["float"] * 1433,
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

We load the graph into memory by calling `g.init()`.

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

The GCN model inherits from the basic learning model class `LearningBasedModel`. As a result, we only need to override the sampling, model construction, and other methods to build GCN model. 

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

`_sample_seed` and `_positive_sample` use to sample seed nodes and positive samples. `_receptive_fn` samples neighbors and organizes `EgoGraph`.
 `OutV` returns one-hop neighbors so the above code samples two-hop neighbors. Users can choose different neighbor sampling methods.
 For the original GCN, it requires all neighbors of each node are so we use 'full' for sampling. We aggregate the sampling results in `EgoGraph` which is the return value.

 ### Graph Data Flow

 In `build` function, we convert `EgoGraph` to `EgoTensor` using `EgoFlow`. `EgoFlow` contains an data flow iterator and several `EgoTensor`s.

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

We can get the `EgoTensor` corresponding to the previous `EgoGraph` from `EgoFlow`.

### Encoder

Next, we first use the feature encoder to encode the original features.
In this example, we use `IdentityEncoder`, that is, return to itself, because
the features of cora are already in vector formats.
For the both discrete and continuous features, we can use `WideNDeepEncoder`,
To learn more encoders, please refer to `python/model/tf/encoders/feature_encoder.py`.
Then, we use the `GCNConv` layer to construct the graph encoder. Each node of GCN samples all neighbors, and the neighbors are organized in a sparse format. Therefore, we use
`SparseEgoGraphEncoder`. For neighbor-aligned model, please refer to the implementation of GraphSAGE.

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

### Loss Function and Training Process

For the Cora node classification model, we can select the corresponding classification loss function in TensorFlow. 
Then, we combine the encoder and loss function in the `build` function, and finally return a data iterator and a loss function.

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

Next, we use `LocalTFTrainer` to train on a single-machine.

```python
def train(config, graph)
  def model_fn():
	return GCN(graph,
			   config['class_num'],
			   config['features_num'],
			   config['batch_szie'],
			   ...)
  trainer = gl.LocalTFTrainer(model_fn, epoch=200)
  trainer.train()

def main():
	config = {...}
	g = load_graph(config)
	g.init(server_id=0, server_count=1, tracker='../../data/')
	train(config, g)
```

This concludes building a GCN model. Please refer to the examples/GCN directory for the complete code.

We have implemented GCN, GAT, GraphSage, DeepWalk, LINE, TransE, Bipartite GraphSage, sample-based GCN and GAT models, which can be used as a starting point for building a similar model.