Commonly used models, including GNNs.

There are 2 types of GNN models:
1. `EgoGraph` based models whose basic data structure is a `EgoGraph`
   which is consits of ego and it's fixed-size neighbors. We implement `EgoGNN`
   to represent these models.
2. `BatchGraph` based models whose basic data structure is a `BatchGraph` 
   which consists of a batch of `SubGraph`s induced from targets and neighbors 
   using `FullNeighborSampler`. Here we implement basic models like homogeneous
    `GraphSAGE`, `GCN`, `GAT`, `SEAL`.
