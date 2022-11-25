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

import os
import sys

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    try:
        import graphlearn.python.nn.pytorch
        from .pytorch.gcn.gcn import GCN as TorchGCN
    except Exception:
        pass

    try:
        import graphlearn.python.nn.tf

        from .tf.trainer import LocalTrainer, DistTrainer

        # backwards compatibility
        LocalTFTrainer = LocalTrainer
        DistTFTrainer = DistTrainer

        from .tf.bipartite_sage.bipartite_sage import BipartiteGraphSAGE
        from .tf.bipartite_sage.hetero_edge_inducer import HeteroEdgeInducer
        from .tf.ego_bipartite_sage.ego_bipartite_sage import EgoBipartiteGraphSAGE
        from .tf.ego_gat.ego_gat import EgoGAT
        from .tf.ego_rgcn.ego_rgcn import EgoRGCN
        from .tf.ego_rgcn.ego_rgcn_data_loader import EgoRGCNDataLoader
        from .tf.ego_sage.ego_sage import EgoGraphSAGE
        from .tf.ego_sage.ego_sage_data_loader import EgoSAGESupervisedDataLoader, \
            EgoSAGEUnsupervisedDataLoader
        from .tf.sage.edge_inducer import EdgeInducer
        from .tf.seal.edge_cn_inducer import EdgeCNInducer
        from .tf.ultra_gcn.ultra_gcn import UltraGCN
    except:  # noqa: E722, pylint: disable=bare-except
        pass

finally:
    sys.path.pop(sys.path.index(os.path.join(os.path.dirname(__file__), '..', '..')))
