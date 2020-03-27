# GAT (Graph Attention Network)
## Introduction
The graph attention network is the follow-up work for GCN. GAT incorporates attention mechanism
into GCN. By calculating attention coefficients among nodes, GAT perform a weighted aggregation
of node and its neighbors, which
allows each node to focus on the most relevant neighbors to make
decisions.

## Key points to build a GAT
- sample Egographs
- encode EgoGraphs using multi-head GAT convolutional layers.

Original GAT uses full graph as input, for efficient large-scale training,
we implemente a sample based version of GAT. For sample based GAT, we use dense format (because sampled number of neighbor nodes are fixed, so they can form a dense tensor)
of EgoGraph and for original GAT, we use a sparse format of EgoGraph (we use sparse tensor to deal with unaligned neighbor numbers) is used for batch training.


## How to run
1. Prepare data
- enter data dir `python cora.py`
2. Train and evaluate
    ```shell
    python train_supervised.py
    ```
## Datasets and performance
| Dataset | ACC   |
| ------- | ----- |
| Cora    | ~0.83 |

## Reference paper 
[graph attention networks](https://arxiv.org/pdf/1710.10903.pdf)
