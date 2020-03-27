# LINE(Large-scale Information Network Embedding)
## Introduction
LINE is one of graph embedding methods which preserves both the local and global
graph structures. It samples edges from graph to measure first-order
proximity and caculates second-order proximity between nodes using their neighbors.

## Key points to build a LINE model
- generate traning pairs.
- encode embedding and caculate their similartiy through two LookupEncoders which store two groups of 
embedding (node embedding and context embedding in the paper)


## How to run
- Prepare data

  enter data dir and `python arxiv.py` to generate data.

- Train

  `python train.py` to train and save embeddings using node2vec.

- Evaluate

  enter eval dir and `python link_prediction_eval.py`

## Dataset and performance

| Dataset | AUC                                |
| ------- | ---------------------------------- |
| arxiv   | ~0.86  (first order)               |
| arxiv   | ~0.72  (second order) batchsize=32 |

## References
[LINE: Large-scale Information Network Embedding](http://www.www2015.it/documents/proceedings/proceedings/p1067.pdf)

[Home](../README.md)
