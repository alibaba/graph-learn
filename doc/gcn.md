# GCN (Graph Convolutional Network)
## Introduction
Graph convolutional network is known as one of the most prominent progress
 for deep learning based method on graph structure data. Convolution operator
 on GCN is a localized first-order approximation of spectral graph convolution.
 In spatial perspective, embeddings of neighbor nodes are aggregated together
 to update node's self embedding.

## Key points to build a GCN
- sample EgoGraphs
- encode EgoGraphs using GCN convolutional layers.

Original GCN uses full graph as input, for efficient large-scale training,
we implemente a sample based version of GCN. For sample based GCN, we use dense format (because sampled number of neighbor nodes are fixed, so they can form a dense tensor)
of EgoGraph and for original GCN, we use a sparse format of EgoGraph (we use sparse tensor to deal with unaligned neighbor numbers) is used for batch training.

## How to run
- Prepare data

  enter data dir `python cora.py`

- Train and evalute 

  `python train_supervised.py` to run the gcn model

## Datasets and performance
| Dataset | ACC   |
| ------- | ----- |
| Cora    | ~0.80 |

## Reference paper 
[semi-supervised classification with graph convolutional networks](https://arxiv.org/abs/1609.02907)

[Home](../README.md)
