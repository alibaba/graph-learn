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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Module(object):
  """ Base class for all neural network modules.

  Your models should inherit from this and implement the `forward()` function.
  Modules can also contain other Modules, allowing to nest them in a tree
  structure.
  ```
    import graphlearn.python.nn.tf as tfg

    class GCN(tfg.Module):
      def __init__(self, layers, active_fn):
        super(GCN, self).__init__()
        self.conv1 = tfg.GCNLayer(layers[0])
        self.conv2 = tfg.GCNLayer(layers[1])
        self.act_fn = active_fn

      def forward(self, x, graph):
        x = self.act_fn(self.conv1(x, graph))
        return self.act_fn(self.conv2(x, graph))
  ```
  """

  def __init__(self):
    pass

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def __str__(self):
    txt = "Module args:\n"
    for k, v in self.__dict__.items():
      txt += "{} : {}\n".format(k, v)
    return txt

  def forward(self, *args, **kwargs):
    raise NotImplementedError
