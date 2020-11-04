#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import graphlearn as gl
import graphlearn.python.graphscope as gs
import tensorflow as tf
import numpy as np

handle = {
    "server": "127.0.0.1:8888,127.0.0.1:8889",
    "client_count": 1,
    "vineyard_id": 78517875468934,
    "node_schema": [
        "person:false:false:2:0:1",
        "software:false:false:1:0:2",
    ],
    "edge_schema": [
        "person:knows:person:false:false:1:1:0",
        "person:created:software:false:false:1:1:0",
    ]
}

def query(g):
    print("Get a batch of nodes...")
    nodes = g.V("person").batch(4).emit()
    print('nodes = ', nodes)
    print(nodes.ids)
    print(nodes.int_attrs)
    print(nodes.float_attrs)
    print(nodes.string_attrs)
    print("Get Nodes Done...")


def test(role, index):
    gl.set_vineyard_graph_id(handle['vineyard_id'])
    gl.set_vineyard_ipc_socket('/tmp/vineyard.sock')
    gl.set_storage_mode(8)

    if role == 'server':
        g = gs.init_graph_from_handle(handle, index)
    else:
        g = gs.get_graph_from_handle(handle, 0, 1)
        query(g)

    g.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('usage: ./demo.py <role("server"/"client")> <index>')
        exit(1)
    role = sys.argv[1]
    index = int(sys.argv[2])
    test(role, index)
