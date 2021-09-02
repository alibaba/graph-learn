![GL](docs/images/graph-learn.png)
[简体中文](README.md) | English

Graph-Learn (formerly AliGraph) is a distributed framework designed for the development and application of large-scale graph neural networks.
It refines and abstracts a set of programming paradigms suitable for common graph neural network models from the practical problems of large-scale graph training, and has been successfully applied to many scenarios such as search recommendation, network security, knowledge graph, etc. within Alibaba.

Graph-Learn provides Python and C++ interfaces for graph sampling operations, and provides a gremlin-like GSL (Graph Sampling Language) interface. For upper layer graph learning models, Graph-Learn provides a set of paradigms and processes for model development, compatible with TensorFlow and PyTorch, providing data layer, model layer interfaces and rich model examples.


[![graph-learn CI](https://github.com/alibaba/graph-learn/workflows/graph-learn%20CI/badge.svg)](https://github.com/alibaba/graph-learn/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alibaba/graph-learn/blob/master/LICENSE)


[**Documentation**](https://graph-learn.readthedocs.io/en/latest/)
## Installation

1. Install Graph-Learn with pip(only for python3)
```
pip install graph-learn
```

2. [Build from source](docs/install_cn.md)

3. Use Docker

4. [k8s](docs/k8s.md)

## example
```
cd examples/tf/ego_sage/
python train_unsupervised.py
```

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
