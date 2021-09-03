## Global Configuration

[nn/tf/config.py](https://github.com/alibaba/graph-learn/tree/master/graphlearn/python/nn)


In model implementation, it is often necessary to distinguish between training and evaluation processes, e.g., dropout, batchnorm, etc. are used differently in training and evaulation phases. In addition, for larger embedding, it is also necessary to consider partitioning them according to the number of ps.
In order in order to facilitate the unified processing of the model layer, we treat these training and configuration-related parameters as a global parameter conf, so that there is no need to distinguish them in the form of multi-layer function passing or placeholder, **just need to directly obtain the values in conf when programming the model, and properly configure these arguments in the training and prediction code.**


#### `conf.training`
The common conf.training is used to distinguish between training and prediction phases, and dropout is not used in the prediction phase. conf.training is already used for control flow when the model is implemented, so you only need to use `conf.training=True` in front of the training process and `conf. training=False` in front of the prediction process.
​

#### `conf.emb_max_partitions`
For the case where embedding is relatively large and requires partitions, you need to set conf.emb_max_partitions to the number of ps in tensorflow, i.e.
`conf.emb_max_partitions=len(FLAGS.ps_hosts.split(','))`
​

Other arguments are configured in the specific implementation of Config, and you can also add your own global configuration.

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
