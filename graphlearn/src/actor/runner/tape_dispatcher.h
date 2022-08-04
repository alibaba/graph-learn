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

#ifndef GRAPHLEARN_ACTOR_RUNNER_TAPE_DISPATCHER_H_
#define GRAPHLEARN_ACTOR_RUNNER_TAPE_DISPATCHER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "brane/actor/actor_implementation.hh"
#include "brane/actor/reference_base.hh"
#include "brane/util/common-utils.hh"
#include "brane/util/data_type.hh"
#include "actor/dag/dag_actor.act.h"
#include "actor/dag/dag_proxy.h"
#include "actor/params.h"
#include "actor/utils.h"
#include "core/dag/tape.h"
#include "platform/env.h"

namespace graphlearn {
namespace act {

class DagActorRefManager;

class TapeDispatcher {
public:
  TapeDispatcher(const std::vector<ActorIdType> *dag_actor_ids);
  virtual ~TapeDispatcher();
  virtual void Dispatch(Tape *tape) = 0;

private:
  void BuildRefs(const std::vector<ActorIdType> *dag_actor_ids);

protected:
  const uint32_t                   local_shards_;
  std::vector<DagActorRefManager*> dag_runner_refs_;
};

std::unique_ptr<TapeDispatcher> NewTapeDispatcher(
  const std::vector<ActorIdType> *dag_actor_ids,
  const DagNode* root);

}  // namespace act
}  // namespace graphlearn

#endif // GRAPHLEARN_ACTOR_RUNNER_TAPE_DISPATCHER_H_
