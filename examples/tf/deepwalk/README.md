# DeepWalk

## Introduction
DeepWalk samples sequences of nodes from the given graph, and use skip-grapm model
to train node embeddings.


## Key points to build a DeepWalk
- sample sequences and generate pairs.
- use LookupEncoders to encode EgoGraphs to node embedding. 

## How to run
1. Prepare data
    ```shell script
    cd ../../data/
    python blogcatelog.py
    ```
2. Train
    ```shell script
    python train.py
    ```
3. Evaluate
    ```shell script
    cd ../../eval/
    python blogcatelog_eval.py
    ```
   to train classfier and get F1 score.


## Dataset and preformance
| Dataset     | macro F1                   |
| ----------- | -------------------------- |
| BlogCatalog | ~0.23  (50% labeled nodes) |

## References
[DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)
