# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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
"""Generate graph data with timestamps
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
cur_path = sys.path[0]

def gen_files():
    import random

    u_count = 20

    with open(os.path.join(cur_path, "data/user"), 'w') as f:
        s = 'id:int64\tweight:float\n'
        f.write(s)
        for i in range(u_count):
            s = '%d\t%f\n' % (i, i / 10.0)
            f.write(s)


    with open(os.path.join(cur_path, "data/u-u"), 'w') as f:
        s = 'src_id:int64\tdst_id:int64\tweight:float\ttimestamp:int64\n'
        f.write(s)
        fanout = 5
        ts = [i for i in range(u_count * fanout)]
        import random
        random.shuffle(ts)
        print(ts)
        for i in range(u_count):
            for j in range(fanout):
                dst = random.randint(0, u_count - 1)
                s = '%d\t%d\t%f\t%d\n' % (i, dst, (i + dst) * 0.1, ts[i * fanout + j])
                f.write(s)

gen_files()
