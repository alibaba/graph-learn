# Bipartite GraphSAGE
## Introduction
Bipartite graphs like user-item graph are very common in e-commerce 
recommendation. We extend GraphSAGE to bipartite graph, called 
bipartite GraphSAGE. 

Here we implement fix-sized `EgoGraph` based bipartite GraphSAGE for u2i 
recommendation which is a two-tower model. We build a GraphSAGE model for 
user and item respectively to generate their respective embedding, and then
calculate the inner product of embeddings.

## How to run
1. Prepare data
```shell script
cd ../../data/
python u2i.py
```

2. Train
```shell script
cd ../tf/ego_bipartite_sage/
python train.py
```