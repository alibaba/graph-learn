# # Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # =============================================================================
# """ Local UT test, run with `sh test_python_ut.sh`.
# """
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import unittest

# import graphlearn as gl
# import numpy as np
# import numpy.testing as npt

# from graphlearn.python.sampler.tests.test_sampling import SamplingTestCase
# import graphlearn.python.tests.utils as utils


# class RandomNodeSubGraphSamplingTestCase(SamplingTestCase):
#   def test_random_node_sampling(self):
#     subgraph_sampler = self.g.subgraph_sampler(seed_type="entity",
#                                                nbr_type="relation",
#                                                batch_size=16,
#                                                strategy="random_node")

#     subgraph = subgraph_sampler.get()
#     row_indices = subgraph.edge_index[0]
#     col_indices = subgraph.edge_index[1]

#     utils.check_subgraph_node_lables(subgraph.nodes)
#     utils.check_subgraph_node_attrs(subgraph.nodes)
#     utils.check_subgraph_edge_indices(subgraph.nodes, row_indices, col_indices)


# if __name__ == "__main__":
#   unittest.main()

# TODO(wenting.swt): check if still needed/