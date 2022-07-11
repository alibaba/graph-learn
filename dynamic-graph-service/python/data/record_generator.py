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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process
import getopt
import numpy as np
import os
import sys
import time
cur_path = sys.path[0]
generated_path = os.path.join(cur_path, "generated")
os.system("mkdir -p " + generated_path)

"""
[TimestampGenerator]s with different distributions.
Supporting for random, zipf and power-law.
"""
class TimestampGenerator(object):
  def __init__(self, low, high, **kwargs):
    self.low = low
    self.high = high

  def next(self):
    pass

class RandomTimestampGenerator(TimestampGenerator):
  def __init__(self, low, high, **kwargs):
    super(RandomTimestampGenerator, self).__init__(low, high)

  def next(self):
    return np.random.randint(self.low, self.high)

class ZipfTimestampGenerator(TimestampGenerator):
  def __init__(self, low, high, **kwargs):
    super(ZipfTimestampGenerator, self).__init__(low, high, kwargs)

  def next(self):
    # TODO(@Seventeen17)
    pass

class PowerTimestampGenerator(TimestampGenerator):
  def __init__(self, low, high, **kwargs):
    super(PowerTimestampGenerator, self).__init__(low, high, kwargs)

  def next(self):
    # TODO(@Seventeen17)
    pass

"""
[RecordGenerator] generate a text file with records.
"""
class RecordGenerator(object):
  def __init__(self,
               v_decoder=None,
               e_decoder=None,
               vtype=0,
               etype=1,
               vbytes=0,
               ebytes=0,
               idcount=100,
               degree=10,
               min_timestamp=10000,
               max_timestamp=10010,
               distribution="Random",
               shuffle_size=20,
               seed=0,
               **kwargs):
    """ Generate records contains both vertices and edges.
    For each id in range [0, idcount), generate #degree vertices of vid=id with
    different properties, and generate #degree of edges of src_vid=id,
    dst_vid=id+degree with different properties. Total size is degree * idcount * 2.

    Schema for vertex and edge can be set in ReocrdGenerator with args \v_decoder and \e_decoder.
    Timestamp is random distribution with low=min_timestamp and high=max_timestamp.
    Zipf and power-low distribution will be supported later.

    Vertex record example: 0(vtype)\t2(vid)\t10010(timestamp)\tb'\x00\x01'(attributes)
    Edge record example: 1(etype)\t2(src_vid)\t3(dst_vid)\t10011(timestamp)\tb'\x00'(attributes)

    args:
      v_decoder: tuple of (TIMESTAMPed, WEIGHTed, LABELed, ATTRIBUTEBytes).
      e_decoder: tuple of (TIMESTAMPed, WEIGHTed, LABELed, ATTRIBUTEBytes).
      idcount: id range.
      degree: for each id, the fanout of vertices and edges.
      min_timestamp: min timestamp.
      max_timestamp: max timestamp.
      distribution: \Random, \Zipf, \Power
      shuffle_size: number of records to shuffle in the buffer,
        when shuffle_size < total size(degree * idcount * 2). shuffle per batch with shuffle_size,
        shuffle_size >= total size, shuffle the whole data,
        shuffle_size == 0, without shuffle.
      seed: random seed.
    """
    # Fixed seed.
    np.random.seed(seed)

    self.vtype = vtype
    self.etype = etype

    self.v_decoder = v_decoder or (True, True, True, vbytes)
    self.e_decoder = e_decoder or (True, True, False, ebytes)

    self.degree = degree
    self.idcount = idcount  # id range

    self.distribution = distribution
    self.min_timestamp = min_timestamp
    self.max_timestamp = max_timestamp

    obj = __import__(self.__module__)
    self.ts_gen = getattr(obj, distribution + "TimestampGenerator")(
      min_timestamp, max_timestamp)

    self.shuffle_size = shuffle_size

  def start(self, task_count):
    # file_name = "records_idcount-{}_degree-{}_mints-{}_maxts-{}_dist-{}_{}".format(
    #     self.idcount, self.degree, self.min_timestamp, self.max_timestamp,
    #     self.distribution, time.time())
    file_name = "records_tmp"
    file_names = [file_name + "_part" + str(task_id) for task_id in range(task_count)]
    # part_size = int(self.idcount / task_count)

    def run(task_id, task_count):
      shuffle_buffer = []
      shuffle_size = self.shuffle_size

      with open(os.path.join(generated_path, file_names[task_id]), 'w') as f:
        for c in range(task_id, self.idcount, task_count):
          for d in range(self.degree):
            if shuffle_size <= 0:
              if shuffle_size > -2:
                np.random.shuffle(shuffle_buffer)
              for rec in shuffle_buffer:
                f.write(rec)
              shuffle_buffer = []
              shuffle_size = self.shuffle_size

            vrecord = "{}\t{}".format(self.vtype, c) # vtype, vid
            vrecord += self.property(self.v_decoder, c)
            shuffle_buffer.append(vrecord)

            e_record = "{}\t{}\t{}".format(self.etype, c, (c + d) % self.idcount) # etype, src_vid, dst_vid
            e_record += self.property(self.e_decoder, c)
            shuffle_buffer.append(e_record)

            shuffle_size -= 2

        np.random.shuffle(shuffle_buffer)
        for rec in shuffle_buffer:
          f.write(rec)
    processes = []
    for i in range(task_count):
      processes.append(Process(target=run, args=(i, task_count)))
    for p in processes:
      p.start()
    for p in processes:
      p.join()
    return file_name, file_names

  def property(self, decoder, i):
    self.timestamp = self.ts_gen.next()
    s = ""
    if decoder[0]:
      s += "\t{}".format(self.timestamp)
    if decoder[1]:
      s += "\t{}".format(i / 10) # weight
    if decoder[2]:
      s += "\t{}".format(i % 2) # label
    if decoder[3] > 0:
      attrs = "a" * decoder[3]
      s += "\t{}".format(attrs) # attributes, with #decoder[3] bytes
    s += '\n'
    return s

def main(argv):
  vtype = 0
  etype = 1
  vbytes = 116
  ebytes = 0
  idcount=100
  degree = 10
  min_timestamp=10000
  max_timestamp=10000 + 100 * 10 * 2
  distribution = "Random"
  shuffle_size = 20
  task_count = 1

  opts, args = getopt.getopt(
    argv, 'v:e:vb:eb:c:d:min:max:s:n', ['vtype=', 'etype=',
      'vbytes=', 'ebytes=', 'idcount=', 'degree=', 'min_timestamp=',
      'max_timestamp=', 'distribution=', 'shuffle_size=', 'task_count='])
  for opt, arg in opts:
    if opt in ('-v', '--vtype'):
      vtype = int(arg)
    elif opt in ('-e', '--etype'):
      etype = int(arg)
    elif opt in ('-vb', '--vbytes'):
      vbytes = int(arg)
    elif opt in ('-eb', '--ebytes'):
      ebytes = int(arg)
    elif opt in ('-c', '--idcount'):
      idcount = int(arg)
    elif opt in ('-d', '--degree'):
      degree = int(arg)
    elif opt in ('-min', '--min_timestamp'):
      min_timestamp = int(arg)
    elif opt in ('-max', '--max_timestamp'):
      max_timestamp = int(arg)
    elif opt in ('-s', '--shuffle_size'):
      shuffle_size = int(arg)
    elif opt in ('-n', '--task_count'):
      task_count = int(arg)
    else:
      pass
  assert(min_timestamp < max_timestamp)
  generator = RecordGenerator(vtype=vtype,
                              etype=etype,
                              vbytes=vbytes,
                              ebytes=ebytes,
                              idcount=idcount,
                              degree=degree,
                              min_timestamp=min_timestamp,
                              max_timestamp=max_timestamp,
                              distribution=distribution,
                              shuffle_size=shuffle_size)
  file_name, file_names = generator.start(task_count=task_count)

  print(file_name)
  # for f in file_names:
  #   print("{} records start from min timestamp {} with random distribution has generated in file \'{}\'."
  #       .format(degree * idcount * 2 / task_count, min_timestamp, f))

if __name__ == '__main__':
  start = time.time()
  main(sys.argv[1:])
  print("Record generator cost {}s".format(time.time() - start))
