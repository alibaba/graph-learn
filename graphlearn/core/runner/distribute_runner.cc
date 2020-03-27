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

#include "graphlearn/core/runner/distribute_runner.h"
#include "graphlearn/include/config.h"

namespace graphlearn {

std::unique_ptr<OpRunner> GetOpRunner(Env* env, op::Operator* op) {
  std::unique_ptr<OpRunner> ret;
  if (GLOBAL_FLAG(DeployMode) < 1) {
    ret.reset(new OpRunner(env, op));
  } else {
    ret.reset(new DistOpRunner(env, GLOBAL_FLAG(ServerId), op));
  }
  return ret;
}

}  // namespace graphlearn
