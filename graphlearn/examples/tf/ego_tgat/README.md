# TGAT (Temporal Graph Attention)
## Introduction
[Inductive Representation Learning on Temporal Graphs](https://arxiv.org/abs/2002.07962)

ref: https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs

Fix-sized `TemporalGraph` based TGAT:
1. Iterates events with asscending timestamps.
2. Samples topk-timestamp neighbors with timestamps constraint for src, dst and neg_dst in the events.
3. Learn src_emb, pos_dst_emb and neg_dst_emb with TGAT by encoding timespan and features together.
4. Train with positive link and negative link with binary cross entropy loss.

## How to run
### Link prediction.
1. Prepare data
    ```shell
    cd examples/data
    python jodie.py
    ```
2. Train and Test
    ```shell
    cd ../tf/ego_tgat/
    python train_eval.py
    ```
## Datasets and performance
| Dataset | Test ACC   |
| ------- | ----- |
| Wikipedia | ~0.8  |