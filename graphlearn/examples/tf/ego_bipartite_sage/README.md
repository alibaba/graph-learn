# Bipartite GraphSAGE
## Introduction
Bipartite graphs like user-item graph are very common in e-commerce
recommendation. We extend GraphSAGE to bipartite graph, called
bipartite GraphSAGE.

Here we implement fix-sized `EgoGraph` based bipartite GraphSAGE for u2i
recommendation which is a two-tower model. We build a GraphSAGE model for
user and item respectively to generate their respective embedding, and then
calculate the inner product of embeddings.

## How to run
1. Prepare data. AmazonBooks for example.
```shell script
cd ../../data/
python amazon_books_data.py
```

2. Train and save embedding.
```shell script
cd ../tf/ego_bipartite_sage/
python train.py
```

3. Evaluate.
```shell script
cd ../../eval
python test_rec.py # need to change file path.
```

4. Result

|Recall@20 |NDCG@20 |Hitrate@20  |
|----      |----    |----        |
|0.0254    |0.0626  |0.1683      |

