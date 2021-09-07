# GCN(Graph Convolutional Network)

## Introduction
Sampling-based GCN based on pyG.

We first use GSL 1-hop full neighbor sampling to get the 1-hop neighbor 
of nodes, and then implement an induce_func to convert the GSL results
to a list of pyG `Data` objects. Then, we use the `PyGDataLoader` to merge the list 
of `Data` to pyG `Batch` object. Finally, we implement the sampling-based 
`GCN` based on pyG `GCNConv`.

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
    cd ../pytorch/gcn/
    python train.py
    ```
3. Training with pytorch DDP
  ```
  python -m torch.distributed.launch --use_env train.py --ddp
  ```