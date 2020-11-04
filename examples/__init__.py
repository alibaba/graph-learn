# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
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
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    import graphlearn.python.model
    import graphlearn.python.model.tf

    from .tf.bipartite_graphsage.bipartite_graph_sage import BipartiteGraphSage
    from .tf.deepwalk.deepwalk import DeepWalk
    from .tf.gat.gat import GAT
    from .tf.gcn.gcn import GCN
    from .tf.graphsage.graph_sage import GraphSage
    from .tf.line.line import LINE
    from .tf.transe.trans_e import TransE
finally:
    sys.path.pop(sys.path.index(os.path.join(os.path.dirname(__file__), "..")))
