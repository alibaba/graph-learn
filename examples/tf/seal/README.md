# SEAL
## Introduction
[Link Prediction Based on Graph Neural Networks](https://papers.nips.cc/paper/2018/file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf)

## How to run
1. Prepare data.
    ```shell script
    cd ../../data/
    python ogbl_collab.py
    ```

2. Training example.

    ```shell script
    cd ../tf/seal/
    python seal_link_predict.py
    ```
## Datasets and performance
| Dataset | hits@50   |
| ------- | ----- |
| ogbl_collab    | ~0.49  |