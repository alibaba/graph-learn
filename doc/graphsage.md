# GraphSAGE (Graph SAmple and aggreGatE)
## Introduction
GraphSAGE is a general inductive framework that efficiently generates node embeddings for previously unseen data. To make use of large-scale graph data, GraphSAGE proposes to sample the computational sub-graphs from original graph data then use a batch by batch training strategy. It also proposes several aggregators to aggregate neighbors' embeddings selectively.

## How to build a GraphSAGE model
- Sample EgoGraphs by sample functions.
- Use EgoFlow to convert EgoGraphs to EgoTensors.
- Use EgoGraphEncoder to encode EgoTensors.
- Feed ecoded embeddings to loss funciton and training.

## How to run
- Prepare Data
  - python data/cora/preprocess.py

- Training
  - supervised: python train_supervised.py
  - unsupervised: python train_unsupervised.py

- Evaluation on ppi dataset

  - enter data dir and `python ppi.py` to generate ppi data.

  - `python train_unsupervised.py` to train and save embeddings.

  - enter eval dir to train and evaluate using
    `python ppi_eval.py`

- Distributed training with TensorFlow.

  **GL** provides a query and sampling service. It has two role, 
  client and server, the client sends requests of query and sampling 
  and server responds and returns the result.

  Suppose you have two machines with ip1 and ip2, each contains
  a TF worker and ps, so there is 2 workers and 2 parameter servers in total.
  The same number of clients and servers of **GL** 
  are co-placed with the workers and parameter-servers. To start 
  **GL** service, a distributed file systme like NFS is needed, you
  should pass this path through the flags `--tracker`.

  To run a distributed example, firstly you should prepare data,
  run with `python cora.py` to generate local data on each machine,
  and then on machine with ip1, run with 

    ```shell script
    python dist_train.py \
      --ps_hosts=ip1:2222,ip2:2222 \
      --worker_hosts=ip1:2223,ip2:2223 \
      --job_name=ps \
      --task_index=0
    
    python dist_train.py \
      --ps_hosts=ip1:2222,ip2:2222 \
      --worker_hosts=ip1:2223,ip2:2223 \
      --job_name=worker \
      --task_index=0
    ````

    and on machine with ip2, run

    ```shell script
     python dist_train.py \
       --ps_hosts=ip1:2222,ip2:2222 \
       --worker_hosts=ip1:2223,ip2:2223 \
       --job_name=ps \
       --task_index=1
     
     python dist_train.py \
       --ps_hosts=ip1:2222,ip2:2222 \
       --worker_hosts=ip1:2223,ip2:2223 \
       --job_name=worker \
       --task_index=1
     
    ```

- Distributed training with k8s.

    see [trian with k8s](k8s.md)
    
## Datasets and performance
### supervised model
| Dataset | ACC   |
| ------- | ----- |
| cora    | ~0.83 |
### unsupervised model
| Dataset | micro F1 score |
| ------- | -------------- |
| ppi     | ~0.49          |

## Reference paper
[Inductive Representation Learning on Large Graphs](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)

[Home](../README.md)
