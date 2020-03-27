# DeepWalk

## Introduction
DeepWalk samples sequences of nodes from the given graph, and use skip-grapm model
to train node embeddings.


## Key points to build a DeepWalk
- sample sequences and generate pairs.
- use LookupEncoders to encode EgoGraphs to node embedding. 

## How to run
- Prepare data

  enter data dir and `python blogcatelog.py` to generate data.

- Train

  `python train.py` to train and save embeddings.

- Evaluate

  enter eval dir and `python blogcatelog_eval.py`
  to train classfier and get F1 score.


## Dataset and preformance
| Dataset     | macro F1                   |
| ----------- | -------------------------- |
| BlogCatalog | ~0.23  (50% labeled nodes) |

## References
[DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)

[Home](../README.md)
