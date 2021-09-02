![GL](docs/images/graph-learn.png)
简体中文 | [English](README_en.md)

Graph-Learn(原AliGraph) 是面向大规模图神经网络的研发和应用而设计的一款分布式框架。
它从大规模图训练的实际问题出发，提炼和抽象了一套适合于常见图神经网络模型的编程范式， 并已经成功应用在阿里巴巴内部的诸如搜索推荐、网络安全、知识图谱等众多场景。

Graph-Learn提供了图采样操作的Python和C++接口，并且提供了一个类似gremlin的GSL(Graph Sampling Language)接口。对于上层图学习模型，Graph-Learn提供了一套模型开发的范式和流程，兼容TensorFlow和PyTorch，提供了数据层，模型层接口和丰富的模型示例。

[![graph-learn CI](https://github.com/alibaba/graph-learn/workflows/graph-learn%20CI/badge.svg)](https://github.com/alibaba/graph-learn/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alibaba/graph-learn/blob/master/LICENSE)



## 安装部署

1. 通过wheel包安装
```
pip install graphlearn
```

2. [从源码编译](docs/install_cn.md)

3. 使用Docker

4. [k8s](docs/k8s.md)

## 运行示例
```
cd examples/tf/ego_sage/
python train_unsupervised.py
```

## 用户文档

### [快速开始](docs/quick_start_cn.md)

### 1. 图操作接口

*  [数据源](docs/data_loader_cn.md)
*  [图对象](docs/graph_object_cn.md)
*  [数据对象](docs/data_object_cn.md)
*  [图查询](docs/graph_query_cn.md)
*  [图遍历](docs/graph_traverse_cn.md)
*  [图采样](docs/graph_sampling_cn.md)
*  [负采样](docs/negative_sampling_cn.md)
*  [**GSL**](docs/gsl_cn.md)

### 2. 算法开发接口
  
*  [范式与流程](docs/gnn_programming_model_cn.md)

*  TensorFlow
    - [数据层](docs/tf_data_layer_cn.md)
    - [模型层](docs/tf_model_layer_cn.md)
    - [常见loss](docs/tf_loss_cn.md)
    - [全局配置](docs/tf_config_cn.md)
    - [模型示例](docs/tf_model_example_cn.md)
    - [开发指南](docs/tf_custom_model_tutorial_cn.md)

*  PyTorch/PyG

    - [开发流程](docs/torch_custom_model_tutorial_cn.md)


### 3. [系统配置](docs/global_config_cn.md)

### 4. 功能扩展

* [自定义算子](docs/operator.md)

* [数据源接入](docs/other_source.md)

## 论文

如果**Graph-Learn**对你的工作有所帮助，请引用如下论文。

```
@article{zhu2019aligraph,
  title={AliGraph: a comprehensive graph neural network platform},
  author={Zhu, Rong and Zhao, Kun and Yang, Hongxia and Lin, Wei and Zhou, Chang and Ai, Baole and Li, Yong and Zhou, Jingren},
  journal={Proceedings of the VLDB Endowment},
  volume={12},
  number={12},
  pages={2094--2105},
  year={2019},
  publisher={VLDB Endowment}
}
```

## 协议

Apache License 2.0.
