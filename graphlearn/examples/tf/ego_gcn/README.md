# [GCN](https://arxiv.org/abs/1609.02907)
## Introduction
Here we implement fix-sized neighbor sampling based GCN. 

## How to run
### Node classification
Here we use cora as an example,

1. Prepare data
```shell script
cd ../../data/
python cora.py
```

2. Train
```shell script
cd ../tf/ego_gcn/
python train_supervised.py
```