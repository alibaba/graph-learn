from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
cur_path = sys.path[0]
tracker = os.path.join(cur_path, "tracker")

def gen_files():
    import random

    u_count = 100
    i_count = 10

    with open(os.path.join(cur_path, "data/user"), 'w') as f:
        s = 'id:int64\tweight:float\n'
        f.write(s)
        for i in range(u_count):
            s = '%d\t%f\n' % (i, i / 10.0)
            f.write(s)

    with open(os.path.join(cur_path, "data/item"), 'w') as f:
        s = 'id:int64\tfeature:string\n'
        f.write(s)
        for i in range(100, 100 + i_count):
            s = '%d\t%s:%d:%f:%f:%s\n' % (i, str(i) + 's', i, i*1.0, i * 10.0, 'x')
            f.write(s)

    with open(os.path.join(cur_path, "data/u-i"), 'w') as f:
        s = 'src_id:int64\tdst_id:int64\tweight:float\n'
        f.write(s)
        for i in range(u_count):
            for j in range(100, 100 + i_count):
                s = '%d\t%d\t%f\n' % (i, j, (i + j) * 0.1)
                f.write(s)

    with open(os.path.join(cur_path, "data/u-i-train"), 'w') as f:
        s = 'src_id:int64\tdst_id:int64\tweight:float\n'
        f.write(s)
        for i in range(u_count):
            for j in range(100, 100 + i_count // 2):
                s = '%d\t%d\t%f\n' % (i, j, (i + j) * 0.1)
                f.write(s)

    with open(os.path.join(cur_path, "data/entity"), 'w') as f:
      s = 'id:int64\tlabel:int64\tfeature:string\n'
      f.write(s)
      for i in range(120):
        s = '%d\t%d\t%f:%f:%f:%f\n' % (i, i, i * 0.1, i * 0.2, i * 0.3, i * 0.4)
        f.write(s)

    with open(os.path.join(cur_path, "data/relation"), 'w') as f:
      s = 'src_id:int64\tdst_id:int64\tweight:float\n'
      f.write(s)
      for i in range(100):
        s = '%d\t%d\t%f\n' % (i, i + 2, (i / 100.0))
        f.write(s)
        s = '%d\t%d\t%f\n' % (i, i + 3, (i / 100.0))
        f.write(s)
        s = '%d\t%d\t%f\n' % (i, i + 5, (i / 100.0))
        f.write(s)

    with open(os.path.join(cur_path, "data/cond_node"), 'w') as f:
      s = 'id:int64\tweight:float\tfeature:string\n'
      f.write(s)
      for i in range(200):
        s = '%d\t%f\t%d:%d:%f:%s\n' % (i, i * 0.1, i % 5, i % 4, i * 0.3, str(i%3))
        f.write(s)

gen_files()
