简体中文 | [English](README_en.md)

![GL](docs/images/graph-learn.png)

[![graph-learn CI](https://github.com/alibaba/graph-learn/workflows/graph-learn%20CI/badge.svg)](https://github.com/alibaba/graph-learn/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alibaba/graph-learn/blob/master/LICENSE)

# 介绍

**Graph-Learn(GL，原AliGraph)** 是面向大规模图神经网络的研发和应用而设计的一款分布式框架，
它从实际问题出发，提炼和抽象了一套适合于当下图神经网络模型的编程范式，
并已经成功应用在阿里巴巴内部的诸如搜索推荐、网络安全、知识图谱等众多场景。

**GL**注重**可移植**和**可扩展**，对于开发者更为友好，为了应对**GNN**在工业场景中的多样性和快速发展的需求。
基于**GL**，开发者可以实现一种**GNN**算法，或者面向实际场景**定制化**一种图算子，例如图采样。
**GL**的接口以Python和NumPy的形式提供，可与TensorFlow或PyTorch兼容但不耦合。
目前**GL**内置了一些结合TensorFlow开发的经典模型，供用户参考。
**GL**可运行于Docker内或物理机上，支持单机和分布式两种部署模式。


# 用户文档

## [**安装部署**](https://github.com/alibaba/graph-learn/wiki/%E5%AE%89%E8%A3%85%E9%83%A8%E7%BD%B2)

## [**快速开始**](https://github.com/alibaba/graph-learn/wiki/%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)

## [**图操作接口**](https://github.com/alibaba/graph-learn/wiki/%E6%95%B0%E6%8D%AE%E6%BA%90)

## [**GNN模型开发**](https://github.com/alibaba/graph-learn/wiki/%E8%8C%83%E5%BC%8F%E4%B8%8E%E6%B5%81%E7%A8%8B)


# 论文

如果**GL**对你的工作有所帮助，请引用如下论文。

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

# 协议

Apache License 2.0。

# 致谢

**GL**孵化于阿里巴巴内部，由计算平台事业部-PAI团队、新零售智能引擎-智能计算实验室、安全部-数据与算法团队共同研发。
研发过程中收到很多有价值的反馈，代码也依赖了以下开源社区的优秀项目，一并感谢。

*  [protobuf](https://github.com/protocolbuffers/protobuf)

*  [grpc](https://github.com/grpc/grpc)

*  [glog](https://github.com/google/glog)

*  [googletest](https://github.com/google/googletest)

*  [TensorFlow](https://github.com/tensorflow/tensorflow)

*  [pybind11](https://github.com/pybind/pybind11)


如果你在使用**GL**过程中遇到什么问题，请留言或发信至graph-learn@list.alibaba-inc.com，也欢迎贡献代码。
