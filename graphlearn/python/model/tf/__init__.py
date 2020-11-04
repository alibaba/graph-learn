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

import os
import sys

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

    # tf based model
    from graphlearn.python.model.tf import aggregators
    from graphlearn.python.model.tf import encoders
    from graphlearn.python.model.tf import layers
    from graphlearn.python.model.tf import utils

    from graphlearn.python.model.tf.trainer import *
    from graphlearn.python.model.tf.optimizer import *
    from graphlearn.python.model.tf.loss_fn import *
    from graphlearn.python.model.tf.ego_tensor import *
    from graphlearn.python.model.tf.ego_flow import *

except ImportError:
   pass
finally:
    sys.path.pop(sys.path.index(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
