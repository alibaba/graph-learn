from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
cur_path = sys.path[0]

def gen_files():
  uid_range = (0, 10000)
  iid_range = (10000, 20000)
  fanout = 15

  with open(os.path.join(cur_path, "data/user"), 'w') as f:
    for i in range(uid_range[0], uid_range[1]):
      # id weight attributes
      s = "{} {} {}\n".format(i, 0.1, str(i))
      f.write(s)

  with open(os.path.join(cur_path, "data/item"), 'w') as f:
    for i in range(iid_range[0], iid_range[1]):
      # id weight attributes
      s = "{} {} {}\n".format(i, 0.1, str(i))
      f.write(s)

  with open(os.path.join(cur_path, "data/u2i"), 'w') as f:
    for i in range(uid_range[0], uid_range[1]):
      for j in range(fanout):
      # sid did weight attributes
        s = "{} {} {} {}\n".format(i, (i + j) % 10000 + 10000, 0.1, str(i))
        f.write(s)

  with open(os.path.join(cur_path, "data/i2i"), 'w') as f:
    for i in range(iid_range[0], iid_range[1]):
      for j in range(fanout):
      # sid did weight attributes
        s = "{} {} {} {}\n".format(i, (i + j) % 1000 + 10000, 0.1, str(i))
        f.write(s)

gen_files()