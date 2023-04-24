# Node2Vec
## Reference
Node2vec: Scalable Feature Learning for Networks

## How to run
1. Prepare data
    ```shell script
    cd ../../data/
    python blogcatelog.py
    ```
2. Local train
    ```shell script
    cd ../tf/node2vec/
    python train.py
    ```
3. Evaluate
    ```shell script
    cd ../../eval
    python blogcatelog_eval.py
    ```

## Dataset and preformance

| Dataset     | macro F1                   |
| ----------- | -------------------------- |
| BlogCatalog | ~0.24  (50% labeled nodes) |
