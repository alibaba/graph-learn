/* Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef GRAPHLEARN_CONTRIB_KNN_BUILDER_H_
#define GRAPHLEARN_CONTRIB_KNN_BUILDER_H_

#include <string>
#include "core/graph/storage/node_storage.h"
#include "include/index_option.h"

namespace graphlearn {

bool BuildKnnIndex(io::NodeStorage* storage, const IndexOption& option);

}  // namespace graphlearn

#endif  // GRAPHLEARN_CONTRIB_KNN_BUILDER_H_
