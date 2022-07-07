## 全局配置

对应[nn/tf/config.py](../../../../python/nn/tf/config.py)


在模型实现中，往往需要区分training和evaluation过程，比如dropout, batchnorm等在training和evaulation阶段的用法不一样。此外，对于比较大的embedding，还需要考虑根据ps个数对其进行partition。<br />
为了方便模型层的统一处理，我们将这些训练和配置相关的参数当成一个全局的参数conf，这样就不用通过多层函数传参或者placeholder的形式来区分，**只需要在模型编程时直接获取conf里的值，在训练、预测代码里正确配置这些参数值即可。**


#### `conf.training`
常见的conf.training用来区别训练和预测阶段，预测阶段不会进行dropout。模型实现时，已经使用了conf.training来进行control flow，使用时只需要在训练过程前面`conf.training=True`，在预测过程前面`conf.training=False`即可。<br />
​

#### `conf.emb_max_partitions`
对于embedding比较大需要partition的情况，需要设置conf.emb_max_partitions为tensorflow的ps数目，即
`conf.emb_max_partitions=len(FLAGS.ps_hosts.split(','))`
​

其他参数配置见Config的具体实现，你也可以新增你自己的一些全局的配置。<br />

```python
class Config(object):
  def __init__(self):
    """Set and get configurations for tf models.

    Configurations:
      training (bool): Whether in training mode or not. Defaults to True.
      emb_max_partitions (int): The `max_partitions` for embedding variables
        partitioned by `min_max_varaible_partitioner`. Defaults to None means
        use ps count as max_partitions.
      emb_min_slice_size (int): The `min_slice_size` for embedding variables
        partitioned by `min_max_varaible_partitioner`. Defaults to 128K.
      emb_live_steps (int): Global steps to live for inactive keys in embedding
        variables. Defaults to None.
    """
    self.training = True
    self.emb_max_partitions = None
    self.emb_min_slice_size = 128 * 1024
    self.emb_live_steps = None

conf = Config()
```
