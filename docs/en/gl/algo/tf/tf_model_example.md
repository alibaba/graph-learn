## Examples

examples/tf


### EgoGraph based GNNs examples
For EgoGraph based GNNs, we provide three algorithm examples, ego_sage, ego_gat, ego_bipartite_sage.
- **ego_sage**: homogeneous graph sage, two training examples are provided for supervised node classification and unsupervised i2i recommendation.
- **ego_gat**:homogeneous gat, provides supervised node classification examples.
- **ego_bipartite_sage**: bipartite graph sage, provides training examples for unsupervised u2i recommendation.

The implementation of EgoGraph based GNN generally just needs to compose different `EgoLayers` with different `EgoConv` and then pass them to `EgoGNN`.The construction of the whole model is similar to building blocks, and the difference between different models is mainly the difference of `EgoConv`.  Let's take `EgoGraphSAGE` as an example to illustrate.


#### EgoGraphSAGE

```python
class EgoGraphSAGE(tfg.EgoGNN):
  def __init__(self,
               dims,
               agg_type="mean",
               bn_func=None,
               act_func=tf.nn.relu,
               dropout=0.0,
               **kwargs):
    """ EgoGraph based GraphSAGE.

    Args:
      dims: An integer list, in which two adjacent elements stand for the
        input and output dimensions of the corresponding EgoLayer. The length
        of `dims` is not less than 2. If len(`dims`) > 2, hidden layers will 
        be constructed. 
        e.g. `dims = [128, 256, 256, 64]`, means the input dimension is 128 and
        the output dimension is 64. Meanwhile, 2 hidden layers with dimension 
        256 will be added between the input and output layer.
      agg_type: A string, aggregation strategy. The optional values are
        'mean', 'sum', 'max' and 'gcn'.
    """
    assert len(dims) > 2

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


We use `EgoSAGEConv` to compose `EgoLayer` and then pass it to `EgoGNN` to quickly build an `EgoGraphSAGE`.

### SubGraph based GNNs(experimental)
For SubGraph based GNNs, we provide two algorithm examples, sage and seal.Both examples start from edge traversal, perform 1-hop full neighbor sampling and negative sampling, and then use reduce_func to obtain SubGraph. SubGraph currently only supports homogeneous graphs.


- **sage**: GraphSAGE, uses BCE loss. uses default reduce_func
- **seal**: SEAL algorithm for link prediction, using a custom induce_func for node labeling.

See nn/tf/model for the model implementation part of SubGraph.


### Distributed example
In order to facilitate the writing of the distributed training process, we have simply wrapped the distributed training process as `DistTrainer`
​

#### DistTrainer
[examples/tf/trainer.py](../../../../examples/tf/trainer.py)

```python
class DistTrainer(object):
  """Class for distributed training and evaluation

  Args:
    cluster_spec: TensorFlow ClusterSpec.
    job_name: name of this worker.
    task_index: index of this worker.
    worker_count: The number of TensorFlow worker.
    ckpt_dir: checkpoint dir.
  """
  def __init__(self,
               cluster_spec,
               job_name,
               task_index,
               worker_count,
               ckpt_dir=None)
   
  # The model definitions must all be placed under the context of DistTrainer 
  def context(self)

  # Training process, training epochs times, will print the training speed and the corresponding loss, at the end of training all workers will
  # Use our implementation of SyncBarrierHook to do a synchronous wait.
  def train(self, iterator, loss, learning_rate, epochs=10, **kwargs)
    
  # The above interface is provided to the worker node of tensorflow, the ps node just needs to execute join
  def join(self)
```
In `DistTrainer` we have simply wrapped a few common functions, you can extend or add new functions to control your own training process.

**Note:**
**TensorFlow distinguishes between ps and worker roles, GraphLearn distinguishes between client and server, the examples we provide are in GraphLearn's server mode.
That is, client and server are distinguished, and generally tf's ps and GraphLearn's server are deployed in the same process, and tf's worker and GraphLearn's client are deployed in the same process.
Therefore, when executing GraphLearn's graph.init, you need to distinguish between different machine roles.**


**DistTrainer provides context interface, model related definitions need to be put under context, in order to place variables to ps automatically. Except the join function, all other functions are provided to worker, please pay attention to the correct use when calling.**
​