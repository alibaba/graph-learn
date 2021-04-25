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
"""utils for ut test, include data generator and result assertion.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import numpy.testing as npt

TRACKER_PATH = '.tracker_path/'
DATA_PATH = '.data_path/'

WEIGHTED = 0
LABELED = 1
ATTRIBUTED = 2
ATTR_TYPES = ['int', 'float', 'string', ('string', 10)]
ENTITY_ATTR_TYPES = ['float', 'float', 'float', 'float']
COND_ATTR_TYPES = ['int', 'int', 'float', 'string']

def prepare_env():
  os.system('mkdir -p %s' % TRACKER_PATH)
  os.system('rm -rf %s*' % TRACKER_PATH)
  os.system('mkdir -p %s' % DATA_PATH)
  os.system('rm -rf %s*' % DATA_PATH)


def gen_node_data(id_type, id_range, schema, mask=""):
  def write_meta(f, schema):
    meta = 'id:int64'
    if WEIGHTED in schema:
      meta += '\tweight:float'
    if LABELED in schema:
      meta += '\tlabel:int64'
    if ATTRIBUTED in schema:
      meta += '\tfeature:string'
    meta += '\n'
    f.write(meta)

  def write_data(f, value, schema):
    line = '%d' % value
    if WEIGHTED in schema:
      line = '%s\t%f' % (line, value / 10.0)
    if LABELED in schema:
      line = '%s\t%d' % (line, value)
    if ATTRIBUTED in schema:
      attr = '%d:%f:%d:%s' % (value, value / 1.0, value, 'hehe')
      line = '%s\t%s' % (line, attr)
    line += '\n'
    f.write(line)

  path = '%s/%s_%d_%s' % (DATA_PATH, id_type, int(time.time() * 1000), mask)
  with open(path, 'w') as f:
    write_meta(f, schema)
    for value in range(id_range[0], id_range[1]):
      write_data(f, value, schema)
  return path


def gen_edge_data(src_type, dst_type, src_range,
                  dst_range, schema, mixed=False, func=None, mask=""):
  if not func:
    func = fixed_dst_ids

  def write_meta(f, schema):
    meta = 'src_id:int64\tdst_id:int64'

    if WEIGHTED in schema:
      meta += '\tweight:float'
    if LABELED in schema:
      meta += '\tlabel:int64'
    if ATTRIBUTED in schema:
      meta += '\tfeature:string'
    meta += '\n'
    f.write(meta)

  def write_data(f, src_type, dst_type, src_id, dst_id, edge_type, schema):
    line = '%d\t%d' % (src_id, dst_id)
    if WEIGHTED in schema:
      line = '%s\t%f' % (line, (src_id + 0.1 * dst_id) / 10.0)
    if LABELED in schema:
      line = '%s\t%d' % (line, src_id)
    if ATTRIBUTED in schema:
      attr = '%d:%f:%d:%s' % (src_id, src_id / 1.0, dst_id, 'hehe')
      line = '%s\t%s' % (line, attr)
    line += '\n'
    f.write(line)

  path = '%s/%s_%s_%d_%s' % (DATA_PATH, src_type,
                          dst_type, int(time.time() * 1000), mask)
  with open(path, 'w') as f:
    write_meta(f, schema)
    src = range(src_range[0], src_range[1])
    # dst = range(dst_range[0], dst_range[1])

    index = 0
    for src_id in src:
      for dst_id in func(src_id, dst_range):
        if index < src_range[1] / 2:
          edge_type = 'first'
        elif not mixed:
          edge_type = 'first'
        else:
          edge_type = 'second'
        write_data(f, src_type, dst_type, src_id, dst_id, edge_type, schema)
        index += 1
  return path


def fixed_dst_ids(src_ids, dst_range):
  if isinstance(src_ids, (int, np.int64)):
    src_ids = [src_ids]
  dst_ids = [src_id * it % (dst_range[1] - dst_range[0]) + dst_range[0] \
             for src_id in src_ids for it in range(1, src_id % 5 + 1)]
  # each src_id has n neighbors, n = src_id % 5
  # so src_id that is multiple pf 5 has no neighbors
  return dst_ids


def gen_entity_node(node_type):
  def write_meta(f):
    meta = 'id:int64\tlabel:int64\tfeature:string\n'
    f.write(meta)

  def write_data(f):
    for i in range(120):
      line = '%d\t%d\t%f:%f:%f:%f\n' % (i, i, i * 0.1, i * 0.2, i * 0.3, i * 0.4)
      f.write(line)

  path = '%s/%s_%d' % (DATA_PATH, node_type, int(time.time() * 1000))
  with open(path, 'w') as f:
    write_meta(f)
    write_data(f)
  return path


def gen_relation_edge(edge_type):
  def write_meta(f):
    meta = 'src_id:int64\tdst_id:int64\tweight:float\n'
    f.write(meta)

  def write_data(f):
    for i in range(100):
      line = '%d\t%d\t%f\n' % (i, i + 2, (i / 100.0))
      f.write(line)
      line = '%d\t%d\t%f\n' % (i, i + 3, (i / 100.0))
      f.write(line)
      line = '%d\t%d\t%f\n' % (i, i + 5, (i / 100.0))
      f.write(line)

  path = '%s/%s_%d' % (DATA_PATH, edge_type, int(time.time() * 1000))
  with open(path, 'w') as f:
    write_meta(f)
    write_data(f)
  return path


def gen_cond_node(node_type):
  def write_meta(f):
    meta = 'id:int64\tweight:float\tfeature:string\n'
    f.write(meta)

  def write_data(f):
    for i in range(200):
      line = '%d\t%f\t%d:%d:%f:%s\n' % (i, i * 0.1, i % 5, i % 4, i * 0.3, str(i%3))
      f.write(line)

  path = '%s/%s_%d' % (DATA_PATH, node_type, int(time.time() * 1000))
  with open(path, 'w') as f:
    write_meta(f)
    write_data(f)
  return path


def check_node_ids(nodes, ids):
  assert set(list(nodes.ids.reshape(-1))).issubset(list(ids))


def check_ids(ids, expected_ids):
  assert set(list(ids.reshape(-1))).issubset(list(expected_ids))


def check_node_type(nodes, node_type=None):
  npt.assert_equal(nodes.type, node_type)


def check_node_shape(nodes, shape):
  npt.assert_equal(nodes.ids.reshape(-1).size, shape)


def check_node_weights(nodes):
  check_id_weights(nodes.ids, nodes.weights)

def check_id_weights(ids, weights):
  npt.assert_almost_equal(weights.reshape(-1), 0.1 * ids.reshape(-1), decimal=5)


def check_not_exist_node_weights(nodes):
  npt.assert_equal(nodes.weights.flatten(), nodes.ids.size * [0.0])


def check_node_labels(nodes):
  npt.assert_equal(nodes.labels, nodes.ids)


def check_not_exist_node_labels(nodes):
  npt.assert_equal(nodes.labels.flatten(), nodes.ids.size * [-1])


def check_node_attrs(nodes):
  size = nodes.ids.size
  check_i_attrs(nodes.int_attrs, nodes.ids)
  check_f_attrs(nodes.float_attrs, nodes.ids)
  check_s_attrs(nodes.string_attrs, nodes.ids)

def check_i_attrs(i_attrs, ids):
  for i, value in zip(range(ids.size), ids):
    npt.assert_equal(i_attrs[i][0], value)

def check_f_attrs(f_attrs, ids):
  for i, value in zip(range(ids.size), ids):
    npt.assert_almost_equal(f_attrs[i][0], value / 1.0, decimal=5)

def check_s_attrs(s_attrs, ids):
  for i, value in zip(range(ids.size), ids):
    npt.assert_equal(s_attrs[i][0], str(value))

def check_not_exist_node_attrs(nodes,
                               default_int_attr=0,
                               default_float_attr=0.0,
                               default_string_attr=""):
  size = nodes.ids.size
  npt.assert_equal([size, 1],
                   list(nodes.int_attrs.shape))  # [batch_size, int_num]
  npt.assert_equal([size, 1],
                   list(nodes.float_attrs.shape))  # [batch_size, float_num]
  npt.assert_equal([size, 1],
                   list(nodes.string_attrs.shape))  # [batch_size, string_num]

  if len(nodes.shape) == 2:
    total_node = nodes.shape[0] * nodes.shape[1]
  else:
    total_node = size

  # the second int is hash value, here we just check the first one
  check_default_i_attrs(nodes.int_attrs, total_node, default_int_attr)
  check_default_f_attrs(nodes.float_attrs, total_node, default_float_attr)
  check_default_s_attrs(nodes.string_attrs, total_node, default_string_attr)

def check_default_i_attrs(i_attrs, count, default_int_attr=0):
  npt.assert_equal(i_attrs[:, 0].flatten(),
                   np.array([default_int_attr] * count))

def check_default_f_attrs(f_attrs, count, default_float_attr=0.0):
  npt.assert_almost_equal(f_attrs.flatten(),
                          np.array([default_float_attr] * count),
                          decimal=4)

def check_default_s_attrs(s_attrs, count, default_string_attr=""):
  npt.assert_equal(s_attrs.flatten(),
                   np.array([default_string_attr] * count))

def check_edge_shape(edges, batch_size):
  npt.assert_equal(edges.src_ids.reshape(-1).size, batch_size)


def check_edge_type(edges, src_type=None,
                    dst_type=None, edge_type=None):
  npt.assert_equal(edges.src_type, src_type)
  npt.assert_equal(edges.dst_type, dst_type)
  npt.assert_equal(edges.edge_type, edge_type)


def check_fixed_edge_dst_ids(edges, dst_range,
                             expected_src_ids=None,
                             default_dst_id=0):
  src_ids = edges.src_ids.reshape(-1)
  dst_ids = edges.dst_ids.reshape(-1)
  batch_size = src_ids.size
  for i in range(batch_size):
    assert src_ids[i] in expected_src_ids
    if src_ids[i] % 5 == 0:
      assert dst_ids[i] == default_dst_id
    else:
      assert dst_ids[i] in fixed_dst_ids(src_ids[i], dst_range)


def check_topk_edge_ids(edges, expected_src_ids,
                        dst_range, expand_factor=2,
                        default_dst_id=0, padding_mode="replicate"):
  src_ids = edges.src_ids.reshape(-1)
  dst_ids = edges.dst_ids.reshape(-1)
  assert set(list(src_ids)) == set(list(expected_src_ids))
  expected_dst_ids = []
  for src_id in expected_src_ids:
    if src_id % 5 == 0:
      expected_dst_ids.append(default_dst_id)
    else:
      expected_part_dst_ids = fixed_dst_ids(src_id, dst_range)
      expected_part_dst_ids.sort(reverse=True)
      real_count = min(src_id % 5, expand_factor)
      if padding_mode == "replicate":
        expected_dst_ids.extend(expected_part_dst_ids[: real_count])
        expected_dst_ids.extend(
            [default_dst_id for i in range(real_count, expand_factor)])
      else:
        for i in range(0, expand_factor, real_count):
          expected_dst_ids.extend(expected_part_dst_ids[: real_count])
        expected_dst_ids.extend(expected_part_dst_ids[: expand_factor % real_count])
  npt.assert_equal(dst_ids, expected_dst_ids)


def check_edge_weights(edges):
  check_src_dst_weights(edges.weights, edges.src_ids, edges.dst_ids)

def check_src_dst_weights(weights, src_ids, dst_ids):
  npt.assert_almost_equal(weights,
                          0.1 * (src_ids + 0.1 * dst_ids),
                          decimal=5)

def check_not_exist_edge_weights(edges):
  if len(edges.shape) == 2:
    total_edge = edges.shape[0] * edges.shape[1]
  else:
    total_edge = edges.size
  npt.assert_almost_equal(edges.weights,
                          np.array([0.0] * total_edge) \
                              .reshape(edges.shape), decimal=5)

def check_half_exist_edge_weights(edges, default_dst_id=0):
  weights = edges.weights.reshape(-1)
  src_ids = edges.src_ids.reshape(-1)
  dst_ids = edges.dst_ids.reshape(-1)
  for src_id, dst_id, weight in zip(src_ids, dst_ids, weights):
    if default_dst_id == dst_id:
      npt.assert_almost_equal(weight, 0.0, decimal=5)
    else:
      npt.assert_almost_equal(weight, 0.1 * (src_id + 0.1 * dst_id), decimal=5)

def check_edge_labels(edges):
  npt.assert_equal(edges.labels, edges.src_ids)

def check_not_exist_edge_labels(edges):
  if len(edges.shape) == 2:
    total_edge = edges.shape[0] * edges.shape[1]
  else:
    total_edge = edges.size
  npt.assert_equal(edges.labels,
                   np.array([-1] * total_edge).reshape(edges.shape))


def check_edge_attrs(edges):
  shape_size = len(edges.shape)
  if shape_size == 2:
    npt.assert_equal(edges.int_attrs[:, :, 0],
                    edges.src_ids)
    npt.assert_almost_equal(edges.float_attrs[:, :, 0],
                            np.reshape([i / 1.0 for i in edges.src_ids],
                                      edges.shape),
                            decimal=5)
    npt.assert_equal(edges.string_attrs[:, :, 0],
                    np.reshape([str(j) for j in \
                        edges.dst_ids.flatten()], edges.shape))
  elif shape_size == 1:
    npt.assert_equal(edges.int_attrs[:, 0],
                    edges.src_ids)
    npt.assert_almost_equal(edges.float_attrs[:, 0],
                            np.reshape([i / 1.0 for i in edges.src_ids],
                                      edges.shape),
                            decimal=5)
    npt.assert_equal(edges.string_attrs[:, 0],
                    np.reshape([str(j) for j in \
                        edges.dst_ids.flatten()], edges.shape))
  else:
    raise NotImplementedError("`check_edge_attrs` only supports edges"
                              " with dim 2 or 3.")

def check_not_exist_edge_attrs(edges,
                               default_int_attr=0,
                               default_float_attr=0.0,
                               default_string_attr=""):
  if len(edges.shape) == 2:
    total_edge = edges.shape[0] * edges.shape[1]
  else:
    total_edge = edges.size

  npt.assert_equal(edges.int_attrs.flatten(),
                   np.array([default_int_attr] * total_edge * 2))
  npt.assert_almost_equal(edges.float_attrs.flatten(),
                          np.array([default_float_attr] * total_edge),
                          decimal=4)
  npt.assert_equal(edges.string_attrs.flatten(),
                   np.array([default_string_attr] * total_edge))


def check_equal(lhs, rhs):
  npt.assert_equal(list(lhs), list(rhs))

def check_val_equal(lhs, rhs):
  npt.assert_equal(lhs, rhs)

def check_sorted_equal(lhs, rhs):
  """ check sorted lhs is equal with sorted rhs.
  """
  lhs = list(lhs)
  rhs = list(rhs)
  lhs.sort()
  rhs.sort()
  npt.assert_equal(lhs, rhs)


def check_subset(a, b):
  """ check a is subset of b.
  """
  assert set(list(a)).issubset(list(b))

def check_disjoint(a, b):
  """ check a is disjoint with b.
  """
  assert set(list(a)).isdisjoint(set(list(b)))

def check_set_equal(lhs, rhs):
  """ check set of lhs is equal with set of rhs.
  """
  npt.assert_equal(set(lhs), set(rhs))

def check_subgraph_node_lables(nodes):
  npt.assert_equal(nodes.labels, nodes.ids)

def check_subgraph_node_attrs(nodes):
  size = nodes.ids.size
  for i, value in zip(range(size), nodes.ids):
    npt.assert_almost_equal(nodes.float_attrs[i][0], value * 0.1, decimal=5)
    npt.assert_almost_equal(nodes.float_attrs[i][1], value * 0.2, decimal=5)
    npt.assert_almost_equal(nodes.float_attrs[i][2], value * 0.3, decimal=5)
    npt.assert_almost_equal(nodes.float_attrs[i][3], value * 0.4, decimal=5)

def check_subgraph_edge_indices(nodes, row_indices, col_indices):
  size = nodes.ids.size
  true_row_idx = []
  true_col_idx = []
  for i in range(size):
    for j in range(size):
      if ((nodes.ids[i] < 100 or nodes.ids[j] < 100)
          and (abs(nodes.ids[i] - nodes.ids[j]) in [2, 3, 5])):
        true_row_idx.append(i)
        true_col_idx.append(j)
  npt.assert_equal(row_indices, np.array(true_row_idx))
  npt.assert_equal(col_indices, np.array(true_col_idx))

