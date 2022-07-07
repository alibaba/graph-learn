# Recommendation.
## Introduction
Example of HeteroSubGraph based unsupervised bipartite GraphSAGE.

Here we take an 'u-i' bipartite graph as exmpale to show how to deal with 
heterogeneous graph using `HeteroSubGraph`. First, we induce the `HeteroSubGraph` 
from the 1-hop full neighbor sampler query. The induced heterogeneous subgraph 
contains two types of nodes: 'u', 'i' and two types of edges ('u', 'u-i', 'i'), 
('i', 'u-i_reverse', 'u'). Then, we use `HeteroConv` to handle the convolution 
on this `HeteroSubGraph`, the base layer of `HeteroConv` is a biparite `SAGEConv`.
Please refer to the code for details.

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