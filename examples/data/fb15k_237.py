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
"""Download preprocessed FB15k-237, used for TransE.
We use the same train/val/test split as the authors.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import download, extract


if __name__ == "__main__":
  download('http://graph-learn-dataset.oss-cn-zhangjiakou.aliyuncs.com/FB15k-237.zip', 'FB15k-237.zip')
  extract('FB15k-237.zip', 'FB15k-237')
