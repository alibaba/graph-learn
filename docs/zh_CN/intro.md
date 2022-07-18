# Introduction

Graph-Learn(原AliGraph) 是面向大规模图神经网络的研发和应用而设计的一款分布式框架。
它从大规模图训练的实际问题出发，提炼和抽象了一套适合于常见图神经网络模型的编程范式， 并已经成功应用在阿里巴巴内部的诸如搜索推荐、网络安全、知识图谱等众多场景。
Graph-Learn1.0版本之后，我们在Graph-Learn训练框架的基础上，增加了在线推理服务，提供了GNN在实际业务中使用的包含训练、推理在内的完整解决方案。

- 训练框架GraphLearn Training

  支持批图上的采样、GNN模型的离线训练与增量训练。

  它提供了图采样操作的Python和C++接口，并且提供了一个类似gremlin的GSL(Graph Sampling Language)接口。对于上层图学习模型，Graph-Learn提供了一套模型开发的范式和流程，兼容TensorFlow和PyTorch，提供了数据层，模型层接口和丰富的模型示例。

- 在线推理服务Dynamic Graph Service

  支持在流式更新的动态图上进行实时的采样。

  它提供了在大规模动态图上采样P99 latency20ms的性能保证。在线推理服务的Client端提供了GSL和Tensorflow Model Predict的 Java接口。


一个完整的训练、推理链路的例子如下：
![overview](../images/overview.png)
1. 用户在Web上发起请求（0），通过Client端在动态图上实时采样（1），利用样本作为模型输入，向Tensorflow Model service请求预测结果（3）；
2. 用户动作、预测结果和反馈的标签、以及Web上的一些context数据落盘到Data Hub（0，3），eg，Log Service；
3. 数据更新作为图更新流入动态图采样服务，更新图（4）；
4. GraphLearn Training小时级别的加载增量数据构图，增量训练模型，部署到tensorflow Model service。

