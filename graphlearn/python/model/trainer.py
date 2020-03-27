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
"""Base class used to build trainers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Trainer(object):
  """ Initialize a base trainer object.

  Args:
    model_func: a function to generate a model.
    epoch: total epoch to be executed
    optimizer: optimization method instance.
  """
  def __init__(self,
               model_func,
               epoch=100,
               optimizer=None):
    self._model_func = model_func

    self._epoch = epoch
    self._optimizer = optimizer

    self._model = None

  def train(self, **kwargs):
    raise NotImplementedError('Trainer.train() to be implemented.')

  def evaluate(self, **kwargs):
    raise NotImplementedError('Trainer.evaluate() to be implemented.')

  def train_and_evaluate(self, **kwargs):
    raise NotImplementedError('Trainer.train_and_evaluate() to be implemented.')

  def get_node_embedding(self, **kwargs):
    raise NotImplementedError('Trainer.get_node_embedding() to be implemented.')

  def get_edge_embedding(self, **kwargs):
    raise NotImplementedError('Trainer.get_edge_embedding() to be implemented.')
