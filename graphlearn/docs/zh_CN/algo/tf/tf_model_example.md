## 模型示例

examples/tf<br/>


### EgoGraph based GNNs示例
EgoGraph based GNNs我们提供了ego_sage, ego_gat, ego_bipartite_sage三个算法示例，

- **ego_sage**: 同构图sage，提供了有监督点分类和无监督i2i推荐两个训练例子。
- **ego_gat**：同构图gat, 提供了有监督点分类例子。
- **ego_bipartite_sage**：二部图sage, 提供了无监督u2i推荐的训练示例。



EgoGraph based GNN的实现一般只需要用不同的`EgoConv`组成不同的`EgoLayer`，再传给`EgoGNN`即可。整个模型的构建类似搭积木过程，不同模型的区别，主要是`EgoConv`的不同。下面我们以`EgoGraphSAGE`为例进行说明<br/>


#### EgoGraphSAGE示例

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


我们使用`EgoSAGEConv`组成`EgoLayer`，然后再传给`EgoGNN`，即可快速搭建出一个`EgoGraphSAGE`。<br/>

### SubGraph based GNNs示例(experimental)
SubGraph based GNNs我们提供了sage和seal两个算法示例。这两个示例都是从边遍历开始，进行1跳的full neighbor sampling和负采样，然后使用induce_func得到SubGraph。SubGraph目前只支持同构图。<br/>

- **sage**： GraphSAGE，使用了BCE loss。使用了默认的induce_func
- **seal**： link prediction的SEAL算法，使用了自定义的induce_func，进行node labeling操作。 <br/>

SubGraph的模型实现部分见nn/tf/model


### 分布式示例
为了方便分布式训练过程的编写，我们对分布式的训练过程简单做了封装，封装成`DistTrainer`
​

#### DistTrainer
对应[examples/tf/trainer.py](../../../../examples/tf/trainer.py)

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
   
  # 模型的定义必须都放到DistTrainer的context下 
  def context(self)

  # 训练过程，训练epochs次，会打印训练速度和对应的loss，训练结束时所有worker会
  # 使用我们实现的SyncBarrierHook做个同步等待。
  def train(self, iterator, loss, learning_rate, epochs=10, **kwargs)
    
  # 上面的接口都是对tensorflow的worker节点提供, ps节点只需要执行join即可
  def join(self)
```
`DistTrainer`里我们简单封装了几个常见的函数，你可以扩展或者新增函数控制你自己的训练过程。
**​**

**注意：**
**TensorFlow区分ps和worker角色，GraphLearn区分client和server，我们提供的示例都是GraphLearn的server mode，也就是client和server区分开，一般是将tf的ps和GraphLearn的server同进程部署，tf的worker和GraphLearn的client同进程部署。因此在执行GraphLearn的graph.init时需要区分不同的机器角色。**
**​**

**DistTrainer提供了context接口，模型相关定义都需要放到context下，便于将variable自动place到ps上，除了join函数，其他的函数都是对worker提供的，请注意调用时正确使用。**
​