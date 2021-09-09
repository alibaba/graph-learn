## 开发指南

### GNNs
#### 快速开始
如果你想快速尝试已经封装的算法，请根据[**examples**](https://github.com/alibaba/graph-learn/tree/master/examples/tf)进行修改，主要需要修改构图时的数据Decoder配置，根据你自己的数据配置合理的参数，以及GSL的图采样逻辑。对EgoGraph based GNNs, 我们提供了**ego_sage**, **ego_gat**, **ego_bipartite_sage**三个示例，对于SubGraph based GNNs, 我们提供了**sage**和**seal**两个示例。详见README.md。
​

### 自定义模型
如果你想自定义GNNs模型，对于EgoGraph来说，仿照examples里的ego_xx算法，你只需要基于**EgoConv**实现自己的Conv层，然后套用**EgoLayer**，构建**EgoGNN**即可，当然你也可以修改EgoLayer和EgoGNN的forward过程。
对SubGraph来说，你需要新增Conv层和对应的模型。
​

### 添加新数据
如果你发现目前的数据格式不能满足你的需求，你需要修改到数据层，包括Dataset, EgoGraph, SubGraph/BatchGraph等接口，将自己需要的数据加进去。我们也在持续完善EgoGraph和SubGraph/BatchGraph，希望能够做到更好的兼容性。
​

此外，对于较为复杂的数据，可以直接使用Data dict的形式，获得query转换后的tensor后自行组织合适的格式。[RGCN](https://github.com/alibaba/graph-learn/tree/master/examples/tf/ego_rgcn)的示例就是通过Data dict实现。
​

### 新增采样算子
如果你需要新增采样算子或者其他图操作算子，需要仿照已经有的sampler进行C++ Op的开发，如有疑问请与我们联系。


​

## 其他图学习模型

对其他图学习模型，比如graph embedding类的deepwalk, node2vec，以及KGs的TransE/DistMult，只需要基于Data dict来编码即可，也就是你需要写一个GSL来描述你的采样过程，然后使用Dataset的`get_data_dict()`接口，得到Data dict，然后可以基于Data dict继续你的模型开发。我们后面也会逐步支持一些这样的算法。
​

​

良好的生态靠大家共同构建和维护，欢迎大家积极贡献！
