## Introduction

End-to-end tutorial about training on vineyard graphs.

## How to run

0. prepare dataset

    ```bash
    $ export GSTEST=/path/to/gstest
    $ git clone https://github.com/GraphScope/gstest.git $GSTEST
    ```

1. starting vineyardd:

    ```bash
    $ export VINEYARD_IPC_SOCKET=/tmp/vineyard.sock
    $ python3 -m vineyard --socket $VINEYARD_IPC_SOCKET
    ```

2. loading graph to vineyard:

    ```bash
    $ vineyard-graph-loader --socket $VINEYARD_IPC_SOCKET --config ./graph.json
    ```

    You will see output likes

    ```
    I0523 11:23:27.517758 1094848 graph_loader.cc:381] [fragment group id]: 3041975930627711
    ```

    Remember the vineyard fragment group id:

    ```bash
    $ export VINEYARD_FRAGMENT_ID=3041975930627711
    ```

3. run the training scripts:

    ```bash
    $ python3 train_supervised.py --vineyard_fragment_id $VINEYARD_FRAGMENT_ID --vineyard_socket $VINEYARD_IPC_SOCKET
    ```

## Hints

0. `PYTHONPATH`

    You may need to setup `PYTHONPATH` properly to make the example script work:

    ```bash
    $ export PYTHONPATH=`pwd`:`pwd`/..:`pwd`/../../..:`pwd`/../../../..
    ```
