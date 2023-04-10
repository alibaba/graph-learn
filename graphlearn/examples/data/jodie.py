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
""" Processor of dynamic graph data Jodie-Wikipedia
from http://snap.stanford.edu/jodie
"""
import os
import pandas as pd
import urllib.request

os.system("mkdir -p jodie")
path = "jodie/wikipedia.csv"
url = 'http://snap.stanford.edu/jodie/wikipedia.csv'
data = urllib.request.urlopen(url)

with open(path, 'wb') as f:
  while True:
    chunk = data.read(10 * 1024 * 1024)
    if not chunk:
      break
    f.write(chunk)

df = pd.read_csv(path, skiprows=1, header=None)

src = df.iloc[:, 0].values
dst = df.iloc[:, 1].values

t = df.iloc[:, 2].values
y = df.iloc[:, 3].values
msg = df.iloc[:, 4:].values

# train:val:test = 0.7:0.15:0.15
val_idx = 110232
test_idx = 133853
max_src_id = 8226
max_dst_id = 9227

def gen_edges(path, start=0, end=-1):
  schema = "sid:int64\tdid:int64\ttimestamp:int64\tattrs:string\n"
  with open(path, 'w') as f:
    f.write(schema)
    for sid, did, ts, attr in zip(
        src[start:end], dst[start:end], t[start:end], msg[start:end]):
      attrs = [str(x) for x in attr.tolist()]
      attrs = ':'.join([str(x) for x in attr.tolist()])
      f.write("{}\t{}\t{}\t{}\n".format(sid, did + max_src_id + 1, int(ts), attrs))

def gen_nodes():
  with open("jodie/src", 'w') as f:
    f.write("id:int64\n")
    for i in range(max_src_id + 1):
      f.write(str(i) + "\n")

  with open("jodie/dst", 'w') as f:
      f.write("id:int64\n")
      for i in range(max_src_id + 1, max_dst_id):
        f.write(str(i) + "\n")

def gen_nodes_withid():
  with open("jodie/src_feat", 'w') as f:
    f.write("id:int64\tattribute:string\n")
    for i in range(max_src_id + 1):
      f.write("{}\t{}\n".format(i, i))

  with open("jodie/dst_feat", 'w') as f:
      f.write("id:int64\tattribute:string\n")
      for i in range(max_src_id + 1, max_dst_id):
        f.write("{}\t{}\n".format(i, i))

gen_edges("jodie/wikipedia")
gen_edges("jodie/wikipedia_train", end=val_idx)
gen_edges("jodie/wikipedia_val", start=val_idx, end=test_idx)
gen_edges("jodie/wikipedia_test", start=val_idx)
gen_nodes()
gen_nodes_withid()
