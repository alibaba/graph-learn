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
"""Base class for all Graph Learning models"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graphlearn.python.model.base_encoder import BaseGraphEncoder


class LearningBasedModel(object):
  """Base class for all learning based models.

  Args:
    graph: Initialized gl.Graph object.
    batch_size: Sample batch_size for training set.
    full_graph_mode: Set True if sample full graph in the first iteration,
    the sampling result will be fixed in the following iterations, usually
    used in inductive models on small scale graph.
  """

  def __init__(self,
               graph,
               batch_size,
               full_graph_mode=False):

    self.graph = graph
    self.batch_size = batch_size
    self.full_graph_mode = full_graph_mode


  def _sample_seed(self):
    """Generator of sample seed using GraphLearn Node/Edge Sampler.
    Returns:
      A generator contains gl.Nodes or gl.Edges.
    """
    raise Exception(" not implemented in base model")

  def _positive_sample(self, t):
    """Sample positive pairs.
    Args:
      t: gl.Nodes or gl.Edges from _sample_seed.
    Returns:
      gl.Edges.
    """
    raise Exception(" not implemented in base model")

  def _negative_sample(self, t):
    """Negative sampling.
    Args:
      t: gl.Edges from _positive_sample.
    Returns:
      gl.Nodes for most of GNN models that do not need edge, or gl.Edges
      for models like TransX that need negative edge to encode.
    """
    raise Exception(" not implemented in base model")

  def _receptive_fn(self, nodes):
    """Get receptive field(neighbors) for Nodes.
    Args:
      nodes: gl.Nodes.
    Returns:
      gl.EgoGraph.
    """
    raise Exception(" not implemented in base model")

  def _encoders(self):
    """
    Returns:
      A dict of encoders which used to encode EgoGraphs into embeddings
    """
    return {"src": BaseGraphEncoder,
            "edge": BaseGraphEncoder,
            "dst": BaseGraphEncoder}

  def build(self):
    """Returns loss and train iterator."""
    raise Exception(" not implemented in base model")

  def val_acc(self):
    """Returns val accuracy and iterator."""
    raise Exception(" not implemented in base model")

  def test_acc(self):
    """Returns test accuracy and iterator."""
    raise Exception(" not implemented in base model")

  def node_embedding(self, type):
    """Returns nodes ids, emb, iterator."""
    raise Exception(" not implemented in base model")

  def edge_embedding(self, type):
    """Returns edges ids, emb, iterator."""
    raise Exception(" not implemented in base model")

  def feed_training_args(self):
    """for other training related settings."""
    return {}

  def feed_evaluation_args(self):
    """for other evaluation related settings."""
    return {}
