Graph-Learn(原AliGraph) 是面向大规模图神经网络的研发和应用而设计的一款分布式框架。
它从大规模图训练的实际问题出发，提炼和抽象了一套适合于常见图神经网络模型的编程范式， 并已经成功应用在阿里巴巴内部的诸如搜索推荐、网络安全、知识图谱等众多场景。

Graph-Learn提供了图采样操作的Python和C++接口，并且提供了一个类似gremlin的GSL(Graph Sampling Language)接口。对于上层图学习模型，Graph-Learn提供了一套模型开发的范式和流程，兼容TensorFlow和PyTorch，提供了数据层，模型层接口和丰富的模型示例。

[![graph-learn CI](https://github.com/alibaba/graph-learn/workflows/graph-learn%20CI/badge.svg)](https://github.com/alibaba/graph-learn/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alibaba/graph-learn/blob/master/LICENSE)

## 安装

```
pip install graphlearn
```

### 从源码编译
```
git clone git@github.com:alibaba/graph-learn.git
cd graphlearn
git submodule update --init
make test
make python
```


## 运行示例
```
cd examples/tf/ego_sage/
python train_unsupervised.py
```

## 协议

Apache License 2.0.
