# TransE
## Introduction
TransE is a popular method to model relationships by interpreting them as translations operating on the
low-dimensional embeddings of the entities. In knowledge graphs, a tuple is in the form of (head, relation, tail). Negative tuples are sampled by corrupting heads or tails in a positive training example.

## Key points to build a TransE
- Get training samples using sample functions.
- Encode entity and relation and use triplet loss to train.

## How to run
1. Prepare Data
    ```shell script
    cd ../../data/
    python fb15k_237.py
    ```

2. Train
    ```shell script
    python train.py
    ```

3. Evaluate
    ```shell script
    python eval.py
    ```

## Datasets and performance
FB15k-237 is a leakage-free version of FB15k, which is usually adopted as one of the benchmark datasets in literature.

| Dataset   | hit@10 |
| --------- | ------ |
| FB15k-237 | 0.41   |

## Reference paper
[Translating Embeddings for Modeling Multi Relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)
