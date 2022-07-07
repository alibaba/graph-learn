# UltraGCN

## How to run
1. Prepare data. AmazonBooks for example.
```shell script
cd ../../data/
python amazon_books_data.py
```

2. Train and save embedding.
```shell script
cd ../tf/ultra_gcn/
python train.py
```

3. Evaluate.
```shell script
cd ../../eval
python test_rec.py # need to change file path.
```

4. Result

10 epochs

|Recall@20 |NDCG@20 |Hitrate@20  |
|----      |----    |----        |
|0.0354   |0.0785   |0.2082      |

