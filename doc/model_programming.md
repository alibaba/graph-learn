# Graph Learning model

## Overview
Learning on graph commonly following two ways. 
The first is to deal with all nodes/edges on the 
whole graph simultaneously, typical examples are the 
original implementation of GCN/GAT, 
which contains an adjacency matrix in its programming model. 
It does not work when the graph becomes larger and larger 
for memory constraints.  The second way is to separate the 
full graph to be multiple subgraphs. 
The training process takes a batch of subgraphs 
in a mini-batch way.


**GL** is mainly designed to deal with extremely 
large scale graph neural network processing. It is a 
combination of a graph engine and upon high-level 
algorithm models. The graph engine stores graph topology data 
and attributes in a distributed way and then provides highly 
efficient sampling interfaces to feed models 
with these data batch-by-batch for training. 

**GL** is a uniform learning framework for different
kinds of models and tasks.
It contains of some common graph learning models such 
as graph embedding models, GNNs, and knowledge graph models. 
The framework is compatible with popular learning frameworks 
like TensorFlow, PyTorch(coming soon).

<p align=center>
<img src="images/learning_model.png"/>
</p>


## Data model

`EgoGraph` is the underlying data model in **GL**. 
It consists of a batch of seed nodes or edges(named 'ego')
with their receptive fields (multi-hops neighbors). 
We implement many build-in samplers to traverse the graph and
sample neighbors. 
Negative samplers are also implemented for unsupervised training.


The sampled data grouped in `EgoGraph` is in numpy format 
to be compatible with different learning framework.
It can be converted to different Tensor format `EgoTensor`
based on the different 
backend. **GL** uses `EgoFlow` to take care of the 
conversion process from `EgoGraph` to `EgoTensor` and pipeline 
the `EgoTensor` for training.

<p align=center>
<img src="images/egograph.png"/>
</p>

## Encoder

All learning based models need encoders to encode `EgoTensor`
to the final embeddings of node, edge or subgraph. 
**GL** first uses feature encoders to encode 
raw features of nodes or edges, then feature embeddings are 
encoded by different models to the final outputs. 
For most of GNNs models, graph encoders provide abstracts on 
how to aggregate neighbors' information to target nodes/edges. 

<p align=center>
<img src="images/egotensor.png"/>
</p>

Based on data model and encoders, different graph learning models
can be easily implemented.


[Home](../README.md)
