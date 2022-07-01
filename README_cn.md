![GL](docs/images/graph-learn.png)
[![pypi](https://img.shields.io/pypi/v/graph-learn.svg)](https://pypi.org/project/graph-learn/)
[![docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://graph-learn.readthedocs.io/zh_CN/latest/)
[![graph-learn CI](https://github.com/alibaba/graph-learn/workflows/graph-learn%20CI/badge.svg)](https://github.com/alibaba/graph-learn/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alibaba/graph-learn/blob/master/LICENSE)

简体中文 | [English](README.md)

Graph-Learn(原AliGraph) 是面向大规模图神经网络的研发和应用而设计的一款分布式框架。
它从大规模图训练的实际问题出发，提炼和抽象了一套适合于常见图神经网络模型的编程范式， 并已经成功应用在阿里巴巴内部的诸如搜索推荐、网络安全、知识图谱等众多场景。

Graph-Learn提供了图采样操作的Python和C++接口，并且提供了一个类似gremlin的GSL(Graph Sampling Language)接口。对于上层图学习模型，Graph-Learn提供了一套模型开发的范式和流程，兼容TensorFlow和PyTorch，提供了数据层，模型层接口和丰富的模型示例。

[**用户文档**](https://graph-learn.readthedocs.io/zh_CN/latest/)

## 安装部署

1. 通过wheel包安装(linux, python3, glibc 2.24+)
```
pip install graph-learn
```

2. [从源码编译](graphlearn/docs/zh_CN/install.md)

3. [使用Docker](graphlearn/docs/zh_CN/install.md)

## 开始使用
GraphSAGE示例
```
cd examples/tf/ego_sage/
python train_unsupervised.py
```
[分布式训练示例](graphlearn/docs/zh_CN/algo/tf/k8s)

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
