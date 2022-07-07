# GAT (Graph Attention Network)
## Introduction
[graph attention networks](https://arxiv.org/pdf/1710.10903.pdf)

Fix-sized `EgoGraph` based GAT.

## How to run
### Supervised node classification.
1. Prepare data
    ```shell script
    cd ../../data/
    python cora.py
    ```
2. Train and evaluate
    ```shell
    cd ../tf/ego_gat/
    python train_supervised.py
    ```
## Datasets and performance
| Dataset | ACC   |
| ------- | ----- |
| Cora    | ~0.8  |