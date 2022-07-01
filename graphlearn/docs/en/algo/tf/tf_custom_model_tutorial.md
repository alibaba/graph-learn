## Development Guide

### GNNs
#### Quick start
If you want to quickly try the already implemented algorithm, please modify it according to [**examples**](https://github.com/alibaba/graph-learn/tree/master/examples/tf), you mainly need to modify the data Decoder configuration for composition, configure reasonable arguments according to your own data and modify the graph sampling of GSL.  For EgoGraph based GNNs, we provide three examples **ego_sage**, **ego_gat**, **ego_bipartite_sage**, and for SubGraph based GNNs, we provide two examples **sage** and **seal**. See README.md for details.
​

### Customizing the model
If you want to customize GNNs model, for EgoGraph, following the ego_xx algorithm in examples, you just need to implement your own Conv layer based on **EgoConv**, then apply **EgoLayer** and build **EgoGNN**, of course you can also modify EgoLayer and EgoGNN's forward process.
For SubGraph, you need to add a new Conv layer and the corresponding model.
​

### Adding new data
If you find that the current data format does not meet your needs, you need to modify to the data layer, including Dataset, EgoGraph, SubGraph/BatchGraph and other interfaces, to add the data you need. We are also continuing to improve EgoGraph and SubGraph/BatchGraph, hoping to achieve better compatibility.

In addition, for more complex data, you can directly use the Data dict form, get the query converted tensor and organize the appropriate format by yourself. The example of [RGCN](https://github.com/alibaba/graph-learn/tree/master/examples/tf/ego_rgcn) is implemented by Data dict.
​
### Adding a sampling operator
If you need to add sampling operators or other graph manipulation operators, you need to follow the already existing sampler for C++ Op development, please contact us if you have any questions.

## Other graph learning models

For other graph learning models, such as deepwalk, node2vec, and KGs TransE/DistMult, you just need to encode them based on Data dict, which means you need to write a GSL to describe your sampling process, and then use Dataset's `get_data_dict()` interface to get Data dict, and then you can continue your model development based on the Data dict. We will gradually support some of these algorithms later as well.
​
