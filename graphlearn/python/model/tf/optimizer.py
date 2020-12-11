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
"""Function of generating TensorFlow optimizers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf


def get_tf_optimizer(name, lr, weight_decay=None):
  """get TensorFlow optimizer.

  Args:
    name: optimizer name
    lr: learning rate
    weight_decay: weight decay rate
  """
  if name == 'adam':
    return tf.train.AdamOptimizer(learning_rate=lr)
  elif name == 'rms_prop':
    return tf.train.RMSPropOptimizer(learning_rate=lr)
  elif name == 'adagrad':
    return tf.train.AdagradOptimizer(learning_rate=lr)
  elif name == 'sgd':
    return tf.train.GradientDescentOptimizer(learning_rate=lr)
  elif name == 'adamW':
    return tf.contrib.opt.AdamWOptimizer(weight_decay=weight_decay,
                                         learning_rate=lr)
  else:
    raise NotImplementedError('Optimizer %s to be implemented.' % name)
