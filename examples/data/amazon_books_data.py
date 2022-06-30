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

import os
import sys
from utils import download, extract

if __name__ == "__main__":
  download('https://graphlearn.oss-cn-hangzhou.aliyuncs.com/data/github/gl_books0.zip', 'gl_books0.zip')
  extract('gl_books0.zip', 'books_data')
  u_count = 52643
  i_count = 91599
  with open(os.path.join(sys.path[0], "books_data/gl_user.txt"), 'w') as f:
    s = 'id:int64\tfeature:string\n'
    f.write(s)
    for i in range(u_count):
      s = '%d\t%s\n' % (i, i)
      f.write(s)

  with open(os.path.join(sys.path[0], "books_data/gl_item.txt"), 'w') as f:
      s = 'id:int64\tfeature:string\n'
      f.write(s)
      for i in range(i_count):
        s = '%d\t%s\n' % (i, i)
        f.write(s)
