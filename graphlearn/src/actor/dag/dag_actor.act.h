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

#ifndef GRAPHLEARN_ACTOR_DAG_DAG_ACTOR_ACT_H_
#define GRAPHLEARN_ACTOR_DAG_DAG_ACTOR_ACT_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hiactor/core/actor-template.hh"
#include "hiactor/util/data_type.hh"

#include "actor/dag/dag_proxy.h"
#include "actor/tensor_map.h"
#include "core/dag/tape.h"
#include "platform/env.h"

namespace graphlearn {
namespace act {

class ANNOTATION(actor:impl) DagActor : public hiactor::actor {
public:
  DagActor(hiactor::actor_base* exec_ctx, const hiactor::byte_t* addr);
  ~DagActor() override;

  seastar::future<hiactor::Void>
  ANNOTATION(actor:method) RunOnce(TapeHolder&& holder);

  ACTOR_DO_WORK()

private:
  ShardableTensorMap*
  BuildInput(const NodeProxy* op, Tape* tape);

  seastar::future<>
  RunInParallel(const NodeProxy* op, ShardableTensorMap* input, Tape* tape);

  seastar::future<JoinableTensorMap*>
  ProcessInShard(const NodeProxy* op, int32_t shard_id, ShardableTensorMap* req);

  bool IsStopping();

private:
  DagProxy dag_proxy_;
  Env*     env_;
  bool     stopping_;
};

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_DAG_DAG_ACTOR_ACT_H_
