# GraphSAGE (Graph SAmple and aggreGatE)
## Introduction
[Inductive Representation Learning on Large Graphs](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)

Fix-sized `EgoGraph` based GraphSAGE.

## How to run
### Supervised node classification.
1. Prepare data
    - cora for supervised model.
    ```shell script
    cd ../../data/
    python cora.py
    ```
2. Train and evaluate

    - supervised: 
    ```shell script
    cd ../tf/ego_sage/
    python train_supervised.py
    ```

### Unsupervised i2i recommendation
1. Prepare data.
    ```shell script
    cd ../../data/
    python ogbl_collab.py
    ```

2. Training example.

    ```shell script
    cd ../tf/ego_sage/
    python train_unsupervised.py
    ```

## Datasets and performance
| Dataset | ACC   |
| ------- | ----- |
| Cora    | ~0.8  |
