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

from graphlearn.python.nn.tf.config import conf
from graphlearn.python.nn.tf.data.dataset import Dataset
from graphlearn.python.nn.tf.data.feature_column import FeatureColumn, \
  EmbeddingColumn, DynamicEmbeddingColumn, NumericColumn, FusedEmbeddingColumn, \
  SparseEmbeddingColumn, DynamicSparseEmbeddingColumn
from graphlearn.python.nn.tf.data.feature_handler import FeatureGroup, \
  FeatureHandler
from graphlearn.python.nn.tf.loss import sigmoid_cross_entropy_loss, \
  unsupervised_softmax_cross_entropy_loss, triplet_margin_loss, triplet_softplus_loss
from graphlearn.python.nn.tf.module import Module

# EgoGraph
from graphlearn.python.nn.tf.data.egograph import EgoGraph
from graphlearn.python.nn.tf.layers.ego_gat_conv import  EgoGATConv
from graphlearn.python.nn.tf.layers.ego_gin_conv import EgoGINConv
from graphlearn.python.nn.tf.layers.ego_layer import EgoLayer
from graphlearn.python.nn.tf.layers.ego_rgcn_conv import EgoRGCNConv
from graphlearn.python.nn.tf.layers.ego_sage_conv import EgoSAGEConv
from graphlearn.python.nn.tf.layers.linear_layer import LinearLayer
from graphlearn.python.nn.tf.model.ego_gnn import EgoGNN
from graphlearn.python.nn.tf.model.link_predictor import LinkPredictor

# SubGraph
from graphlearn.python.nn.tf.data.batchgraph import BatchGraph
from graphlearn.python.nn.tf.data.hetero_batchgraph import HeteroBatchGraph
from graphlearn.python.nn.tf.data.subgraph_inducer import SubGraphInducer
from graphlearn.python.nn.tf.data.subgraph_processor import SubGraphProcessor
from graphlearn.python.nn.tf.layers.gat_conv import GATConv
from graphlearn.python.nn.tf.layers.gcn_conv import GCNConv
from graphlearn.python.nn.tf.layers.hetero_conv import HeteroConv
from graphlearn.python.nn.tf.layers.sage_conv import SAGEConv
from graphlearn.python.nn.tf.layers.sub_conv import SubConv
from graphlearn.python.nn.tf.model.gat import GAT
from graphlearn.python.nn.tf.model.gcn import GCN
from graphlearn.python.nn.tf.model.sage import GraphSAGE
from graphlearn.python.nn.tf.model.seal import SEAL

# utils
from graphlearn.python.nn.tf.utils.compute_norm import compute_norm
from graphlearn.python.nn.tf.utils.softmax import unsorted_segment_softmax
from graphlearn.python.nn.tf.utils.sync_barrier import SyncBarrierHook


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
