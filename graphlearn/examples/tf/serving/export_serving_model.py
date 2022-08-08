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
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import time
from xmlrpc.client import Boolean

import tensorflow as tf
from tensorflow.python.framework import ops, sparse_tensor

logging.basicConfig(level=logging.DEBUG)


def _add_suffix(name):
  return name if len(name.split(':')) > 1 else name + ':0'


class ReplaceRules(object):
  def __init__(self, src=None, target=None):
    self._rules = {}
    if isinstance(target, ops.Tensor):
      if isinstance(src, str):
        self.add(src, target)
      else:
        try:
          getattr(src, "name")
          self.add(str(src.name), target)
        except AttributeError:
          raise Exception('invalid replace src %s and target %s' % (
            str(src), str(target)))
    elif (isinstance(src, sparse_tensor.SparseTensor) and
          isinstance(target, sparse_tensor.SparseTensor)):
      self.add(src.indices.name, target.indices)
      self.add(src.values.name,  target.values)
      self.add(src.dense_shape.name,  target.dense_shape)

  def __str__(self):
    txt = "Replace rules:\n"
    for k, v in self._rules.items():
      txt += "{} -> {}\n".format(k, v)
    return txt

  def add(self, source, target):
    assert(type(source) == str), "`source` must be a str, got {}".format(
      type(source))
    self._rules[_add_suffix(source)] = target

  def count(self):
    return len(self._rules)

  def update(self, rules):
    assert(isinstance(rules, ReplaceRules))
    self._rules.update(rules._rules)

  def get(self, source):
    return self._rules.get(_add_suffix(source), None)


def export_model(ckpt, ph_idxs, saved_model, version, name_prefix, output_tensor):
  """
  Export tf `SavedModel` from checkpoint.
  args:
      ckpt: input checkponit dir
      ph_idxs: list of Placeholder indexes. For Supervised job trained by
      one EgoGraph, all placeholders should be feed when serving. For Unsupervised
      job, for example, u-i link-prediction, there may exists multiple EgoGraphs(u-i-i,
      i-i-i, ...), but only predict user embedding with online tf model serving,
      then only part of placeholders along with u-i-i path will be needed to feed.
  """
  graph = tf.Graph()
  with tf.Session(graph=graph) as sess:
    # Restore from checkpoint
    loader = tf.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
    loader.restore(sess, ckpt)

    inputs = []
    for ph_idx in ph_idxs:
      name = name_prefix + '_' + \
        str(ph_idx) if ph_idx > 0 else name_prefix
      inputs.append(tf.saved_model.utils.build_tensor_info(
        graph.get_operation_by_name(name).outputs[0]))

    output = tf.saved_model.utils.build_tensor_info(
      graph.get_operation_by_name(output_tensor).outputs[0])
    prefix = "IteratorGetNext_ph_input_"
    signature_def_map = {"predict_actions":
                         tf.saved_model.signature_def_utils.build_signature_def(
                           inputs={prefix + str(i): input
                                   for i, input in zip(ph_idxs, inputs)},
                           outputs={output_tensor: output},
                           method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                         )}
    # Export checkpoint to SavedModel
    export_path = os.path.join(tf.compat.as_bytes(saved_model),
                               tf.compat.as_bytes(str(version)))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                         strip_default_attrs=True,
                                         signature_def_map=signature_def_map)
    builder.save()


def modify_graph(input_ckpt, output_ckpt, model_dir, src_node, ph_suffix):
  """
  Modify compute graph trained with GraphLearn tf models for tensorflow model serving.
  Replace the input tensor /IteratorGetNext with mutiple placeholders.
  """
  def replace_inputs(graph, rules):
    def replace_input(node, input_idx, rules):
      target = rules.get(node.inputs[input_idx].name)
      if target is not None:
        node._update_input(input_idx, target)

    logging.info(str(rules))
    for node in graph.get_operations():
      for input_idx in range(len(node.inputs)):
        replace_input(node, input_idx, rules)

  def check_required(graph, node):
    for op in graph.get_operations():
      for input_idx in range(len(op.inputs)):
        op_path = _add_suffix(op.inputs[input_idx].name)
        if op_path == node.name:
          return True
    return False

  output_pb = "graph.pbtxt"

  graph = tf.Graph()
  with tf.Session(graph=graph) as sess:
    # load the original model
    model = tf.train.latest_checkpoint(input_ckpt)

    loader = tf.train.import_meta_graph(model + '.meta', clear_devices=True)
    saver = tf.train.Saver(sharded=True)
    saver.restore(sess, model)
    graph = sess.graph

    # modify graph and return placeholder indexes.
    n_ph = 0
    src_node = graph.get_operation_by_name(src_node)
    rules = ReplaceRules()
    for output in src_node.outputs:
      if check_required(graph, output):
        n_ph += 1
        replace = tf.placeholder(output.dtype, shape=output.shape,
                                 name=output.name.split(":")[0] + ph_suffix)
        rule = ReplaceRules(output, replace)
        rules.update(rule)
    replace_inputs(graph, rules)
    writer=tf.summary.FileWriter('./tensorboard_modified', sess.graph)

    tf.train.write_graph(graph.as_graph_def(),
                         output_ckpt,
                         output_pb,
                         as_text=True)
    saver.save(sess, model_dir)
    writer.close()
  return n_ph


if __name__ == '__main__':
  argparser = argparse.ArgumentParser("Show Compute Graph and Export Serving Model")

  argparser.add_argument('--input_ckpt_dir', type=str, help="Input checkpoint directory")
  argparser.add_argument('--input_ckpt_name', type=str, help="Input checkpoint file name")

  argparser.add_argument('--export', type=Boolean, default=True, help="Export serving model or not")
  argparser.add_argument('--placeholders', type=str, default="",
                         help="Input placeholder indices for serving model with delimiter `,`, \
                         maybe a part of placeholders in modified ckpt, since the others are not \
                         the inputs of current serving model with sub-computational-graph. \
                         You can check it with TensorBoard.")
  argparser.add_argument('--output_model_path', type=str, help="Output serving model path")
  argparser.add_argument('--output_tensor', type=str, default="output_embeddings", help="Predict output tensor name")
  argparser.add_argument('--version', type=int, default=1, help="Serving model version")
  args = argparser.parse_args()

  modified_ckpt = args.input_ckpt_name + '_modified'
  pb = 'graph.pbtxt'
  model_file = "model"
  src_node = "IteratorGetNext"
  placeholder_suffix = "_placeholder"

  name_prefix = src_node + placeholder_suffix
  modified_model = os.path.join(args.input_ckpt_dir, modified_ckpt, model_file)

  print("Start modifing checkpoint to satify model serving...")
  num_placeholders = modify_graph(os.path.join(args.input_ckpt_dir, args.input_ckpt_name),
                                  os.path.join(args.input_ckpt_dir, modified_ckpt),
                                  modified_model,
                                  src_node,
                                  placeholder_suffix)
  print("The modified checkpoint is save as {}.".format(modified_model))

  if args.export:
    print(args.export)
    placeholders = map(int, args.placeholders.split(','))
    export_model(os.path.join(args.input_ckpt_dir, modified_ckpt, model_file),
                 placeholders,
                 args.output_model_path,
                 args.version,
                 name_prefix,
                 args.output_tensor)
    print("Exported serving model in {}.".format(args.output_model_path))
