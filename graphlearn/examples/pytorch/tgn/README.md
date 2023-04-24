# TGN(Temporal Graph Network)

https://arxiv.org/abs/2006.10637

## Introduction
GraphLearn Temporal Sampling-based TGN is implemented based on pyG-tgn.

ref: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/tgn.py

1. Iterates events with timestamp incrementally.
2. amples topk-timestamp neighbor with timestamp constraint for src, dst and neg_dst in the events.
3. Induces the src, dst, neg_dst and neighbors as edge_index with interaction(t, msg) between them.
4. Train with tgn.

## Prepare environment
Install packages in `requirements.txt`, then install the master pyG:

```
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```

## Run
1. Prepare data
```
cd examples/data
python jodie.py
```
2. Train and evaluate
```
cd ../pytorch/tgn
python train_and_eval.py
```

## Results
```
Epoch: 50, Loss: 0.4017, Time: 15.3097s
Val AP: 0.9727, Val AUC: 0.9696
Test AP: 0.9666, Test AUC: 0.9636
```