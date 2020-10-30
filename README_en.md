[简体中文](README.md) | English

![GL](docs/images/graph-learn.png)

[![graph-learn CI](https://github.com/alibaba/graph-learn/workflows/graph-learn%20CI/badge.svg)](https://github.com/alibaba/graph-learn/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alibaba/graph-learn/blob/master/LICENSE)

# Introduction

**Graph-Learn(GL)** is a distributed framework designed for the development and application of large-scale graph neural networks.
It refines and abstracts a set of programming paradigms suitable for the current neural network model.
It has been successfully applied to many scenarios such as search recommendation, network security, and knowledge graphs within Alibaba.


To support the diversity and rapid development of **GNN** in industrial scenarios, **GL** focuses on **portability** and **scalability**, which is more friendly to developers.
Developers can use **GL** to implement a **GNNs** algorithms, or
**customize** a graph operator, such as graph sampling.
The interfaces of **GL** are provided in the form of Python and NumPy. It is compatible with TensorFlow or PyTorch.
Currently, **GL** has some build-in classic models developed with TensorFlow for the user reference.
**GL** can run in Docker or on a physical machine, and supports both stand-alone and distributed deployment modes.

# Documents

* [Installation](docs/install.md)

* [Quick Start](docs/quick_start.md)

* [Conception and API](docs/concept_api.md)

* [System Config](docs/system_config.md)

* [Distribution](docs/distribution.md)

* [Extend](docs/extend.md)

* [Model Examples](docs/model_examples.md)


# Paper

Please cite the following paper in your publications if **GL** helps your research.

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

# License

Apache License 2.0.

# Acknowledgement

**GL** is developed by several teams at Alibaba, including Computing Platform Department - PAI team,
New Retail Intelligence Engine - Data Analytics And Intelligence Lab, and Security Department - Data and Algorithms team.
Thanks to the ones who provide helps and suggestions to open source.

**GL** refers to the following projects. Thanks to the authors and contributors.

*  [protobuf](https://github.com/protocolbuffers/protobuf)

*  [grpc](https://github.com/grpc/grpc)

*  [glog](https://github.com/google/glog)

*  [googletest](https://github.com/google/googletest)

*  [TensorFlow](https://github.com/tensorflow/tensorflow)

*  [pybind11](https://github.com/pybind/pybind11)

Please email graph-learn@list.alibaba-inc.com if any questions. Welcome to contribution!
