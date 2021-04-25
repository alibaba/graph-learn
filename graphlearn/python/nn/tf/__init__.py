# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from graphlearn.python.nn.tf.app.link_predictor import LinkPredictor, \
  SupervisedLinkPredictor, UnsupervisedLinkPredictor
from graphlearn.python.nn.tf.app.node_classifier import NodeClassifier
from graphlearn.python.nn.tf.config import conf
from graphlearn.python.nn.tf.data.data_flow import DataFlow
from graphlearn.python.nn.tf.data.ego_graph import EgoGraph
from graphlearn.python.nn.tf.data.entity import Vertex
from graphlearn.python.nn.tf.data.feature_column import FeatureColumn, \
  EmbeddingColumn, DynamicEmbeddingColumn, NumericColumn, FusedEmbeddingColumn, \
  SparseEmbeddingColumn, DynamicSparseEmbeddingColumn
from graphlearn.python.nn.tf.data.feature_group import FeatureGroup, \
  FeatureHandler
from graphlearn.python.nn.tf.layers.ego_gat_layer import  EgoGATLayerGroup, \
  EgoGATLayer
from graphlearn.python.nn.tf.layers.ego_gin_layer import EgoGINLayerGroup, \
  EgoGINLayer
from graphlearn.python.nn.tf.layers.ego_sage_layer import EgoSAGELayerGroup, \
  EgoSAGELayer
from graphlearn.python.nn.tf.layers.input_layer import InputLayer
from graphlearn.python.nn.tf.layers.linear_layer import LinearLayer
from graphlearn.python.nn.tf.model.ego_gat import EgoGAT, HomoEgoGAT
from graphlearn.python.nn.tf.model.ego_gin import EgoGIN, HomoEgoGIN
from graphlearn.python.nn.tf.model.ego_sage import EgoGraphSAGE, \
  HomoEgoGraphSAGE
from graphlearn.python.nn.tf.module import Module
from graphlearn.python.nn.tf.trainer import Trainer

# subgraph
from graphlearn.python.nn.tf.data.batchgraph import BatchGraph
from graphlearn.python.nn.tf.data.batchgraph_flow import BatchGraphFlow, SubKeys
from graphlearn.python.nn.tf.data.subgraph import SubGraph
from graphlearn.python.nn.tf.data.utils.induce_graph_with_edge import induce_graph_with_edge
from graphlearn.python.nn.tf.layers.gat_conv import GATConv
from graphlearn.python.nn.tf.layers.gcn_conv import GCNConv
from graphlearn.python.nn.tf.layers.sage_conv import SAGEConv
from graphlearn.python.nn.tf.layers.sub_conv import SubGraphConv
from graphlearn.python.nn.tf.loss import softmax_cross_entropy_loss, \
  sigmoid_cross_entropy_loss
from graphlearn.python.nn.tf.model.gat import GAT
from graphlearn.python.nn.tf.model.gcn import GCN
from graphlearn.python.nn.tf.model.sage import GraphSAGE
from graphlearn.python.nn.tf.model.seal import SEAL
from graphlearn.python.nn.tf.utils.compute_norm import compute_norm
from graphlearn.python.nn.tf.utils.softmax import unsorted_segment_softmax

# Special dunders that we choose to export:
_exported_dunders = set([
    '__version__',
    '__git_version__',
    '__compiler_version__',
    '__cxx11_abi_flag__',
    '__monolithic_build__',
])

# Expose symbols minus dunders, unless they are whitelisted above.
__all__ = [s for s in dir() if s in _exported_dunders or not s.startswith('_')]
