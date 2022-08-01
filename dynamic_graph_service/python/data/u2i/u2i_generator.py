# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
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

import argparse
import errno
import numpy as np
import os


user_num = 100
item_num = 1000

feature_num = 10
feature_delim = ":"

training_delim = "\t"
u2i_training_fanout_window = 15
i2i_training_fanout_window = 10

streaming_delim = ","
u2i_streaming_fanout_window = 30
i2i_streaming_fanout_window = 20
streaming_epoch = 1000

output_dir = "/tmp/u2i_gen"


def create_dir(dirname):
  if not os.path.exists(dirname):
    try:
      os.makedirs(dirname)
    except OSError as exc:
      if exc.errno != errno.EEXIST:
        raise


def get_random_weight():
  return "%.3f" % (np.random.rand() + np.random.randint(low=0, high=4))


def get_random_feature():
  features = np.random.rand(feature_num)
  return feature_delim.join([("%.3f" % f) for f in features])


class Vertex(object):
  def __init__(self, vtype, vid, ts=0):
    self.vtype = vtype
    self.vid = vid
    self.feature = get_random_feature()
    self.ts = ts

  def update_feature(self):
    self.feature = get_random_feature()

  def update_timestamp(self, ts):
    self.ts = ts

  def as_training_data_line(self):
    return "{id}{d}{feat}\n".format(
      d=training_delim,
      id=self.vid,
      feat=self.feature)

  def as_streaming_data_line(self):
    return "{vtype}{d}{id}{d}{ts}{d}{feat}\n".format(
      d=streaming_delim,
      vtype=self.vtype,
      id=self.vid,
      ts=self.ts,
      feat=self.feature)


class Edge(object):
  def __init__(self, etype, src_vid, dst_vid, ts=0):
    self.etype = etype
    self.src_vid = src_vid
    self.dst_vid = dst_vid
    self.weight = get_random_weight()
    self.ts = ts

  def as_training_data_line(self):
    return "{src_vid}{d}{dst_vid}{d}{weight}\n".format(
      d=training_delim,
      src_vid=self.src_vid,
      dst_vid=self.dst_vid,
      weight=self.weight)

  def as_streaming_data_line(self):
    return "{etype}{d}{src_vid}{d}{dst_vid}{d}{ts}{d}{weight}\n".format(
      d=streaming_delim,
      etype=self.etype,
      src_vid=self.src_vid,
      dst_vid=self.dst_vid,
      ts=self.ts,
      weight=self.weight)


class U2IGenerator(object):
  def __init__(self):
    self._users = [Vertex("user", i) for i in range(user_num)]
    self._items = [Vertex("item", i) for i in range(item_num)]
    self._u2i_grid_size = int(item_num / user_num)

    self._init_u2i = []
    self._init_i2i = []
    self._create_init_edges()

  def generate_training_data(self):
    training_data_dir = os.path.join(output_dir, "training")
    create_dir(training_data_dir)

    user_file = os.path.join(training_data_dir, "user.txt")
    with open(user_file, 'w') as fw:
      fw.write("id:int64{d}feature:string\n".format(d=training_delim))
      for user in self._users:
        fw.write(user.as_training_data_line())

    item_file = os.path.join(training_data_dir, "item.txt")
    with open(item_file, 'w') as fw:
      fw.write("id:int64{d}feature:string\n".format(d=training_delim))
      for item in self._items:
        fw.write(item.as_training_data_line())

    u2i_file = os.path.join(training_data_dir, "u2i.txt")
    with open(u2i_file, 'w') as fw:
      fw.write("src_id:int64{d}dst_id:int64{d}weight:float\n".format(d=training_delim))
      for u2i in self._init_u2i:
        fw.write(u2i.as_training_data_line())

    i2i_file = os.path.join(training_data_dir, "i2i.txt")
    with open(i2i_file, 'w') as fw:
      fw.write("src_id:int64{d}dst_id:int64{d}weight:float\n".format(d=training_delim))
      for i2i in self._init_i2i:
        fw.write(i2i.as_training_data_line())

  def generate_streaming_data(self):
    streaming_data_dir = os.path.join(output_dir, "streaming")
    create_dir(streaming_data_dir)

    # write pattern
    pattern_file = os.path.join(streaming_data_dir, "u2i.pattern")
    with open(pattern_file, 'w') as fw:
      fw.write("#VERTEX:user{d}#ID{d}timestamp{d}feature\n".format(d=streaming_delim))
      fw.write("#VERTEX:item{d}#ID{d}timestamp{d}feature\n".format(d=streaming_delim))
      fw.write("#EDGE:u2i{d}#SRC:user{d}#DST:item{d}timestamp{d}weight\n".format(d=streaming_delim))
      fw.write("#EDGE:i2i{d}#SRC:item{d}#DST:item{d}timestamp{d}weight\n".format(d=streaming_delim))

    data_file = os.path.join(streaming_data_dir, "u2i.streaming")
    with open(data_file, 'w') as fw:
      # write initial data
      for user in self._users:
        fw.write(user.as_streaming_data_line())
      for item in self._items:
        fw.write(item.as_streaming_data_line())
      for init_u2i in self._init_u2i:
        fw.write(init_u2i.as_streaming_data_line())
      for init_i2i in self._init_i2i:
        fw.write(init_i2i.as_streaming_data_line())
      # write streaming data
      for epoch in range(streaming_epoch):
        ts = epoch + 1

        user_id = np.random.randint(low=0, high=user_num)
        self._users[user_id].update_feature()
        self._users[user_id].update_timestamp(ts)

        item1_id = np.random.randint(
          low=user_id * self._u2i_grid_size,
          high=user_id * self._u2i_grid_size + u2i_streaming_fanout_window
        ) % item_num
        self._items[item1_id].update_feature()
        self._items[item1_id].update_timestamp(ts)
        new_u2i = Edge("u2i", user_id, item1_id, ts)

        item2_id = np.random.randint(
          low=item1_id + 1,
          high=item1_id + 1 + i2i_streaming_fanout_window
        ) % item_num
        self._items[item2_id].update_feature()
        self._items[item2_id].update_timestamp(ts)
        new_i2i = Edge("i2i", item1_id, item2_id, ts)

        fw.write(self._users[user_id].as_streaming_data_line())
        fw.write(self._items[item1_id].as_streaming_data_line())
        fw.write(self._items[item2_id].as_streaming_data_line())
        fw.write(new_u2i.as_streaming_data_line())
        fw.write(new_i2i.as_streaming_data_line())

  def _create_init_edges(self):
    # create u2i edges
    for user_id in range(user_num):
      item_id = user_id * self._u2i_grid_size
      for j in range(u2i_training_fanout_window):
        self._init_u2i.append(Edge("u2i", user_id, item_id))
        item_id = (item_id + 1) % item_num
    # create i2i edges
    for src_item_id in range(item_num):
      dst_item_id = src_item_id + 1
      while dst_item_id <= src_item_id + i2i_training_fanout_window and dst_item_id < item_num:
        self._init_i2i.append(Edge("i2i", src_item_id, dst_item_id))
        dst_item_id += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Data generator for u2i schema')
  parser.add_argument('--user-num', action="store", dest="user_num",
                      help="user number to generate.")
  parser.add_argument('--item-num', action="store", dest="item_num",
                      help="item number to generate")
  parser.add_argument('--feature-num', action="store", dest="feature_num",
                      help="feature number for each vertex(user/item).")
  parser.add_argument('--streaming-epoch', action="store", dest="streaming_epoch",
                      help="epoch for generating streaming data.")
  parser.add_argument('--output-dir', action="store", dest="output_dir",
                      help="The output directory of generated files.")
  args = parser.parse_args()

  if args.user_num is not None:
    user_num = int(args.user_num)
  if args.item_num is not None:
    item_num = int(args.item_num)
  if args.feature_num is not None:
    feature_num = int(args.feature_num)
  if args.streaming_epoch is not None:
    streaming_epoch = int(args.streaming_epoch)
  if args.output_dir is not None:
    output_dir = args.output_dir

  generator = U2IGenerator()
  generator.generate_training_data()
  generator.generate_streaming_data()
