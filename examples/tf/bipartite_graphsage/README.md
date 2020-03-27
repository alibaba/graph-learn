# Bipartite GraphSage
## Introduction
Bipartite graphs are very common in e-commerce recommendation.
We extend GraphSage model to bipartite graph, called Bipartite GraphSage.
The main difference between Bipartie GraphSage and original GraphSage is
that Bipartite GraphSage uses different encoders for source and destination nodes.

## How to run
1. Prepare data
    - enter data dir and `python u2i.py`

2. Train
    - `python train_unsupervised.py` to train and save embeddings.
