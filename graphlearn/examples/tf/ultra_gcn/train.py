# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
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

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

import graphlearn as gl

from ultra_gcn import UltraGCN


def load_graph(config):
  g = gl.Graph()\
        .node("../../data/books_data/gl_item.txt", node_type="i",
              decoder=gl.Decoder(attr_types=['int']))\
        .node("../../data/books_data/gl_user.txt", node_type="u",
              decoder=gl.Decoder(attr_types=['int']))\
        .edge("../../data/books_data/gl_train.txt", edge_type=("u", "i", "u-i"),
              decoder=gl.Decoder(weighted=False), directed=False)\
        .edge("../../data/books_data/gl_i2i.txt", edge_type=("i", "i", "i-i"),
              decoder=gl.Decoder(weighted=True), directed=True)
  return g

def dump_embedding(sess, iter, ids, emb, emb_writer):
  sess.run(iter.initializer)
  while True:
    try:
      outs = sess.run([ids, emb])
      # [B,], [B,dim]
      feat = [','.join(str(x) for x in arr) for arr in outs[1]]
      for id, feat in zip(outs[0], feat):
        emb_writer.write('%d\t%s\n' % (id, feat))
    except tf.errors.OutOfRangeError:
      print('Save node embeddings done.')
      break

def run(config):
  gl.set_shuffle_buffer_size(102400000)
  g = load_graph(config)
  g.init()
  model = UltraGCN(g, config['batch_size'], 
      config['neg_num'], config['neg_sampler'],
      config['user_num'], config['item_num'], config['output_dim'], 
      i2i_weight=config['i2i_weight'], neg_weight=config['neg_weight'], 
      l2_weight=config['l2_weight'])
  train_iter = model.train_iter
  u_iter = model.u_iter
  i_iter = model.i_iter
  loss = model.forward()
  u_ids, u_emb = model.user_emb()
  i_ids, i_emb = model.item_emb()
  optimizer=tf.train.AdamOptimizer(learning_rate=config['learning_rate'])
  train_op = optimizer.minimize(loss)
  train_ops = [loss, train_op]
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(train_iter.initializer)
    print("Start Training...")
    step = 0
    for i in range(config['epochs']):
      try:
        while True:
          ret = sess.run(train_ops)
          if step % 1000 == 0:
            print("Epoch {}, Iter {}, Loss {:.5f}".format(i, step, ret[0]))
          step += 1
      except tf.errors.OutOfRangeError:
        sess.run(train_iter.initializer) # reinitialize dataset.
    print("Start saving embeddings...")
    u_emb_writer = open('u_emb.txt', 'w')
    i_emb_writer = open('i_emb.txt', 'w')
    u_emb_writer.write('id:int64\temb:string\n')
    i_emb_writer.write('id:int64\temb:string\n')
    dump_embedding(sess, u_iter, u_ids, u_emb, u_emb_writer)
    dump_embedding(sess, i_iter, i_ids, i_emb, i_emb_writer)
    u_emb_writer.close()
    i_emb_writer.close()
  g.close()


if __name__ == "__main__":
  config = {'batch_size': 512,
            'user_num': 52643,
            'item_num': 91599,
            'output_dim': 128,
            'neg_num': 10,
            'neg_weight': 10,
            'i2i_weight': 2.75,
            'l2_weight': 1e-4,
            'learning_rate': 1e-3,
            'epochs': 10,
            'neg_sampler': 'random',
            }
  run(config)