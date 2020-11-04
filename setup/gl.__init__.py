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

from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python.config import * # pylint: disable=wildcard-import
from graphlearn.python.graph import Graph
from graphlearn.python.values import Nodes, Edges, Layer, Layers, \
  SparseNodes, SparseEdges
from graphlearn.python.errors import * # pylint: disable=wildcard-import
from graphlearn.python.decoder import Decoder
from graphlearn.python.topology import Topology
from graphlearn.python.sampler import *
from graphlearn.python.graphscope import *

# model
from graphlearn.python.model.base_encoder import *
from graphlearn.python.model.ego_graph import *
from graphlearn.python.model.ego_spec import *
from graphlearn.python.model.learning_based_model import *
from graphlearn.python.model.utils import *
# tf based model
from graphlearn.python.model.tf import aggregators
from graphlearn.python.model.tf import encoders
from graphlearn.python.model.tf import layers
from graphlearn.python.model.tf import utils

from graphlearn.python.model.tf.trainer import *
from graphlearn.python.model.tf.optimizer import *
from graphlearn.python.model.tf.loss_fn import *
from graphlearn.python.model.tf.ego_tensor import *
from graphlearn.python.model.tf.ego_flow import *


EDGE_SRC = pywrap.NodeFrom.EDGE_SRC
EDGE_DST = pywrap.NodeFrom.EDGE_DST
NODE = pywrap.NodeFrom.NODE

REPLICATE = pywrap.PaddingMode.REPLICATE
CIRCULAR = pywrap.PaddingMode.CIRCULAR
