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

from graphlearn.python.nn.pytorch.data.dataset import Dataset
from graphlearn.python.nn.pytorch.data.utils import get_cluster_spec, \
get_counts, launch_server, set_client_num
from graphlearn.python.nn.pytorch.data.pyg_dataloader import PyGDataLoader
from graphlearn.python.nn.pytorch.data.temporal_dataset import TemporalDataset
from graphlearn.python.nn.pytorch.data.temporal_dataloader import TemporalDataLoader


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
