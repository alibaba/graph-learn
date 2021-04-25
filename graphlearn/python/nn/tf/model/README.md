There are 2 types of models:
1. Ego-graph based models like egosage, whose basic data structure is a `EgoGraph`
   which is consits of ego and it's fixed-size neighbors.
2. Sub-graph based models like seal, whose basic data structure is a
   `BatchGraph` consists of a batch of `SubGraph`s which is induced from
   targets and neighbors using `FullNeighborSampler`.
