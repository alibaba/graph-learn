# [RGCN](https://arxiv.org/abs/1703.06103)
## Introduction
RGCN is used to model multi-relational data (multi types of edge) with
GCN. Here we implement fix-sized neighbor sampling based RGCN. 

## How to run
### Node classification
Here we use cora as an example, using two cora's edge tables to simulate 
2 different types of edges(relations).
1. Prepare data
```shell script
cd ../../data/
python cora.py
```

2. Train
```shell script
cd ../tf/ego_rgcn/
python train_supervised.py
```