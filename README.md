![GL](docs/images/graph-learn.png)

# Introduction

**Graph-Learn(GL)** is a framework designed to simplify the application of graph neural networks(**GNNs**).
It abstracts solutions from the actual production cases.
These solutions have been applied and verified on recommendation, anti-cheating and knowledge graph systems.

**GL** is **portable** and **flexible**, which makes it much more friendly to developers.
Based on **GL**, developers are able to implement a kind of **GNNs** algorithms,
**customize** some graph related operators and extend the existed modules easily.
**GL** can be installed in containers or on physical machines, and deployed in single machine mode or **distributed mode**.

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

The developers of **GL** are from several teams at Alibaba, including Computing Platform Department - PAI team,
New Retail Intelligence Engine - Data Analytics And Intelligence Lab, and Security Department - Data and Algorithms team.
Thanks to the ones who provide helps and suggestions to open source.

**Welcome to contribution!**

**GL** refers to the following projects. Thanks to the authors and contributors.

*  [protobuf](https://github.com/protocolbuffers/protobuf)

*  [grpc](https://github.com/grpc/grpc)

*  [glog](https://github.com/google/glog)

*  [googletest](https://github.com/google/googletest)

*  [TensorFlow](https://github.com/tensorflow/tensorflow)

*  [pybind11](https://github.com/pybind/pybind11)
