![GL](docs/images/graph-learn.png)
[![pypi](https://img.shields.io/pypi/v/graph-learn.svg)](https://pypi.org/project/graph-learn/)
[![docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://graph-learn.readthedocs.io/en/latest/)
[![graph-learn CI](https://github.com/alibaba/graph-learn/workflows/graph-learn%20CI/badge.svg)](https://github.com/alibaba/graph-learn/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alibaba/graph-learn/blob/master/LICENSE)

[简体中文](README_cn.md) | English

Graph-Learn (formerly AliGraph) is a distributed framework designed for the development and application of large-scale graph neural networks.
It abstracts a set of programming paradigms suitable for common graph neural network models from the practical problems of large-scale graph training, and has been successfully applied to many scenarios such as search recommendation, network security, knowledge graph, etc. within Alibaba.

Graph-Learn provides both Python and C++ interfaces for graph sampling operations, and provides a gremlin-like GSL (Graph Sampling Language) interface. For upper layer graph learning models, Graph-Learn provides a set of paradigms and processes for model development. It is compatible with TensorFlow and PyTorch, and provides data layer, model layer interfaces and rich model examples.


[**Documentation**](https://graph-learn.readthedocs.io/en/latest/)

## Installation

1. Install Graph-Learn with pip(linux, python3, glibc 2.24+)
```
pip install graph-learn
```

2. [Build from source](graphlearn/docs/en/install.md)

3. [Use Docker](graphlearn/docs/en/install.md)


## Getting Started
GraphSAGE example
```
cd examples/tf/ego_sage/
python train_unsupervised.py
```

[Distributed training example](graphlearn/docs/en/algo/tf/k8s.md)



## Citation

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

## License

Apache License 2.0.
