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

try:
    from .python.version import __version__, __git_version__
except ImportError:
    pass

from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.pywrap_graphlearn import IndexOption

from graphlearn.python.cluster import get_cluster
from graphlearn.python.config import * # pylint: disable=wildcard-import
from graphlearn.python.data import *  # pylint: disable=wildcard-import
from graphlearn.python.errors import * # pylint: disable=wildcard-import
from graphlearn.python.graph import Graph
from graphlearn.python.gsl import Dataset
from graphlearn.python.operator import * # pylint: disable=wildcard-import
from graphlearn.python.sampler import * # pylint: disable=wildcard-import
from graphlearn.python.utils import * # pylint: disable=wildcard-import
import graphlearn.python.nn as nn

EDGE_SRC = pywrap.NodeFrom.EDGE_SRC
EDGE_DST = pywrap.NodeFrom.EDGE_DST
NODE = pywrap.NodeFrom.NODE

REPLICATE = pywrap.PaddingMode.REPLICATE
CIRCULAR = pywrap.PaddingMode.CIRCULAR
