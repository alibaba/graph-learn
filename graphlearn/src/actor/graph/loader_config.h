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

#ifndef GRAPHLEARN_ACTOR_GRAPH_LOADER_CONFIG_H_
#define GRAPHLEARN_ACTOR_GRAPH_LOADER_CONFIG_H_

#include <cstdint>

#include "brane/core/shard-config.hh"
#include "include/config.h"

namespace graphlearn {
namespace act {

struct LoaderConfig {
  // actor_id = 0 << 16 | index; index = {1..3}
  static const uint32_t graph_actor_id   = 1;
  static const uint32_t control_actor_id = 2;
  static const uint32_t sync_actor_id    = 3;
};

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_GRAPH_LOADER_CONFIG_H_
