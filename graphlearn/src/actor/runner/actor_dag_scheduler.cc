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

#include "core/runner/dag_scheduler.h"

#include "boost/range/irange.hpp"
#include "hiactor/core/coordinator.hh"
#include "seastar/core/alien.hh"

#include "actor/dag/dag_actor_manager.h"
#include "actor/runner/tape_dispatcher.h"
#include "common/base/log.h"
#include "core/dag/tape.h"

namespace graphlearn {
namespace act {

class ActorDagScheduler;

void RunnerLoop(const Dag* dag, ActorDagScheduler *self,
                const std::vector<ActorIdType> *dag_actor_ids);

class ActorDagScheduler : public DagScheduler {
public:
  explicit ActorDagScheduler(Env* env) : DagScheduler(env) {}

  ~ActorDagScheduler() override {
    for (auto& run_loop : run_loops_) {
      if (run_loop.joinable()) {
        run_loop.join();
      }
    }
  }

  void Run(const Dag* dag) override {
    auto fut = seastar::alien::submit_to(
        *seastar::alien::internal::default_instance, 0, [dag, this] () mutable {
      return RegisterDAG(dag);
    });
    fut.wait();
    run_loops_.emplace_back(
        RunnerLoop, dag, this, DagActorManager::GetInstance().GetDagActorIds(dag->Id()));
  }

  bool Stop() const {
    return env_->IsStopping();
  }

private:
  seastar::future<> RegisterDAG(const Dag* dag) {
    LOG(INFO) << "Register dag " << dag->Id();
    DagActorManager::GetInstance().AddDag(dag);
    return hiactor::coordinator::get().global_barrier(
        "REGISTER_DAG_" + std::to_string(dag->Id()), false);
  }

private:
  std::vector<std::thread> run_loops_;
};

void RunnerLoop(const Dag* dag, ActorDagScheduler* self,
                const std::vector<ActorIdType>* dag_actor_ids) {
  auto dispatcher = act::NewTapeDispatcher(dag_actor_ids, dag->Root());
  TapeStorePtr store = GetTapeStore(dag->Id());

  if (store == nullptr) {
    LOG(FATAL) << "Dag " << dag->Id() << " hasn't been registered.";
    return;
  }

  while (!self->Stop()) {
    /// The dag will be executed round by round in the background until
    /// the server stops. The results of each running round will be dumped
    /// to a Tape. The capacity of TapeStore is limited to the memory size.
    /// TapeStore can not generate a new tape until some old ones are
    /// consumed by clients.
    Tape* tape = store->New();
    dispatcher->Dispatch(tape);

    store->WaitAndPush(tape, [self] {
      return self->Stop();
    });
  }
}

}  // namespace act

DagScheduler* NewActorDagScheduler(Env* env) {
  return new act::ActorDagScheduler(env);
}

}  // namespace graphlearn
