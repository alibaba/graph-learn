## 开发流程

基于pytorch的GNNs算法开发流程包括如下步骤：

1. 构造点表和边表数据并完成构图
2. 根据需求写GSL query，完成子图采样
3. 使用Dataset和DataLoader处理GSL的数据
4. 编写模型
5. 训练/预测


GraphLearn兼容了社区开源框架pyG([https://github.com/rusty1s/pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)), 因此想基于pyG开发GNNs算法时，在写好GSL后需要实现一个induce_func完成GSL产生的`gl.nn.Data` dict到pyG的`Data`的转换。然后使用GraphLearn封装的`PyGDataLoader`即可以将`Data`合并成`Batch`对象，后面的模型部分直接使用pyG实现即可。

### 数据层
对应nn/pytorch/data

使用GraphLearn编写模型时，首先需要构图，并写GSL图查询query, 将需要采样的meta-path通过GSL描述，然后就可以产生一个返回numpy ndarry的数据流，为了方便模型层使用，GraphLearn实现了pytorch的`Dataset`，来把GSL的query转换成一个tensor格式的`gl.nn.Data`的dict或者是induce成一个pyG的`Data`对象。然后就可以通过pytorch的DataLoader来遍历获取数据。


#### Dataset
Dataset提供了两个基本功能：

- **通用数据转换**：将GSL的数据转换成一个的`gl.nn.Data` dict，并转成tensor格式，然后为了接pytorch的`torch.utils.data.DataLoader`提供了`as_dict`接口，将每个`Data`对象转成一个dict，最终遍历时返回一个大的dict，元素是`Data`转换后的dict。
- **转成pyG的数据**：通过自定义的`induce_func`构造子图，即将numpy的`gl.nn.Data` dict转换成一个大小为batch_size(GSL里指定)的pyG `Data`的list，然后通过`PyGDataLoader`将`Data` list合并成pyG的`Batch`对象。

```python
class Dataset(th.utils.data.IterableDataset):
  def __init__(self, query, window=5, induce_func=None):
    """Dataset reformats the sampled batch from GSL query as `Data` object
    consists of Pytorch Tensors.
    Args:
      query: a GSL query.
      window: the buffer size for query.
      induce_func:  A function that takes in the query result `Data` dict and
        returns a list of subgraphs. For pyG, these subgraphs are pyG `Data`
        objects.
    """
    self._rds = RawDataset(query, window=window)
    self._format = lambda x: x
    self._induce_func = induce_func


  def as_dict(self):
    """Convert each `Data` to dict of torch tensors.
    This function is used for raw `DataLoader` of pytorch.
    """
```


#### PyGDataLoader


为了方便对Dataset induce出的Data list进行合并处理，GraphLearn封装了一个面向pyG的`PyGDataLoader`。**注意**，由于GraphLearn batch的操作在GSL里已经产生，**Dataset的一次迭代返回的已经是一个batch的数据了**，因此在`PyGDataLoader`实现里强制batch_size=1。

```python
class Collater(object):
  def __init__(self):
    pass

  def collate(self, batch):
    batch = batch[0]
    elem = batch[0]
    if isinstance(elem, Data):
      return Batch.from_data_list(batch)

  def __call__(self, batch):
    return self.collate(batch)


class PyGDataLoader(torch.utils.data.DataLoader):
    """pyG Data loader which merges a list of pyG `Data` objects induced
    from a the `graphlearn.python.nn.pytorch.data.Dataset` to a pyG `Batch` object.

    Args:
      dataset (Dataset): The dataset to convert GSL and induce a list of pyG `Data` objects.
    """
    def __init__(self, dataset, **kwargs):
      if "batch_size" in kwargs:
        del kwargs["batch_size"]
      if "collate_fn" in kwargs:
        del kwargs["collate_fn"]
      super(PyGDataLoader, self).__init__(dataset, batch_size=1, collate_fn=Collater(), **kwargs)
```

### 模型层
#### pyG
使用提供了`induce_func`的`Dataset`和`PyGDataLoader`后，返回的数据为pyG的`Batch`对象，因此可以直接复用pyG的模型和层。


#### 其他
如果不想使用pyG，也可以基于Dataset的到的gl.nn.Data的dict对数据进行操作，然后基于pytorch来编写模型即可。如果你有好的建议请与我们联系。


### 示例
完整示例见 [examples/pytorch](https://github.com/alibaba/graph-learn/tree/master/examples/pytorch)。
