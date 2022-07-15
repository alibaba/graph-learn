# Quick Start
Graph-Learn contains two parts, **GraphLearn-Training** and **Dynamic-Graph-Service**.

## GraphLearn-Training

```
cd graphlearn
```

It runs GNN models training job, and then save embedding for offline vector recall, or save checkpoint for online predict with tensorflow model serive.

Here is a quickly start of **standalone** model training.

1. Pull docker image

```
docker pull graphlearn/graphlearn
```

2. update wheel

```
pip install -U graph-learn
```

> You can also build wheel from source, [ref](gl/install.md)

3. train the model

```
cd examples
# Prepare data
cd data && python cora.py
cd ../tf/ego_sage && python train_supervised.py
```

Quickly start **distributed** training mode with kubeflow.

1. Configure yaml

```
cd k8s
vim dist.yaml
# config ${your_job_name}, ${your_namespace},  ${your_volume} and ${your_pvc_name}
```

2. Start job

```
kubectl apply -f dist.yaml
```

>Besides training GNN model, GraphLearn-Training provides multiple layer of apis for describe your own graph, sampler and models, [ref](gl/quick_start.md).

## Dynamic-Graph-Service

```
cd dynamic_graph_service
```

It relies on GraphLearn-Training trained models.
Refer to e2e tutorial. [ref](tutorial.md)
