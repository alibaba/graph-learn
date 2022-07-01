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
"""Download fake u2i data for Bipartite GraphSage example.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import download, extract

if __name__ == "__main__":
  download('https://graphlearn.oss-cn-hangzhou.aliyuncs.com/data/github/u2i.zip', 'u2i.zip')
  extract('u2i.zip', 'u2i')
