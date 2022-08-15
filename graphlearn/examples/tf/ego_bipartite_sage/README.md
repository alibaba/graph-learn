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
### Train with AmazonBooks
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


### Generate u2i and i2i bipartite graph data.

1. Prepare data.
```
cd dynamic_graph_service
python2 python/data/u2i/u2i_generator.py --feature-num=10 --output-dir /tmp/u2i_gen
```

2. Train model and save ckpt.

```
python train.py --user_path="/tmp/u2i_gen/training/user.txt" \
--item_path="/tmp/u2i_gen/training/item.txt" \
--u2i_path="/tmp/u2i_gen/training/u2i.txt" \
--i2i_path="/tmp/u2i_gen/training/i2i.txt" \
--batch_size=64 \
--u_attr_types='["float","float","float","float","float","float","float","float","float","float"]' \
--u_attr_dims='[0,0,0,0,0,0,0,0,0,0]' \
--i_attr_types='["float","float","float","float","float","float","float","float","float","float"]' \
--i_attr_dims='[0,0,0,0,0,0,0,0,0,0]' \
--edge_weighted=True
```
