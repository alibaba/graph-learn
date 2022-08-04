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
#include "brane/actor/actor_client.hh"
#include "brane/actor/actor_message.hh"
#include "brane/actor/actor_param_store.hh"
#include "brane/core/coordinator.hh"
#include "brane/core/shard-config.hh"
#include "brane/util/data_type.hh"
#include "actor/dag/dag_actor_manager.h"
#include "actor/runner/tape_dispatcher.h"
#include "common/base/errors.h"
#include "common/base/log.h"
#include "core/dag/tape.h"
#include "seastar/core/alien.hh"

namespace graphlearn {
namespace act {

struct RegisterMessage {
  const DagActorManager* dag_actor_manager;

  RegisterMessage() = default;
  explicit RegisterMessage(const DagActorManager* manager)
    : dag_actor_manager(manager) {
  }

  // Currently, reister message only passed within the same machine.
  void dump_to(brane::serializable_queue &qu) {  // NOLINT [runtime/references]
  }

  static RegisterMessage load_from(brane::serializable_queue &qu) { // NOLINT [runtime/references]
    return RegisterMessage();
  }
};

class ActorDagScheduler;

void RunnerLoop(const Dag* dag, ActorDagScheduler *self,
                const std::vector<ActorIdType> *dag_actor_ids);

class ActorDagScheduler : public DagScheduler {
public:
  explicit ActorDagScheduler(Env* env) : DagScheduler(env) {
    brane::actor_param_store::get().set_register_func(
        [] (brane::actor_message* msg,
            brane::actor_param_store::param_map& params) {
      auto* message = reinterpret_cast<
        brane::actor_message_with_payload<RegisterMessage>*>(msg);
      auto* manager = message->data.dag_actor_manager;

      auto *dag_actor_ids = manager->GetDagActorIds();
      for (auto dag_actor_id : *dag_actor_ids) {
        params.insert(dag_actor_id, manager->GetActorParams(dag_actor_id));
      }

      for (auto& id_pair : *(manager->GetOpActorIds())) {
        auto op_actor_id = id_pair.second;
        params.insert(op_actor_id, manager->GetActorParams(op_actor_id));
      }
    });
  }

  ~ActorDagScheduler() {
    for (auto &run_loop : run_loops_) {
      run_loop.join();
    }

    for (auto& manager : dag_actor_managers_) {
      delete manager.second;
    }
  }

  void Run(const Dag* dag) override {
    auto fut = seastar::alien::submit_to(0, [dag, this] () mutable {
      return RegisterDAG(dag);
    });
    fut.wait();
    run_loops_.emplace_back(RunnerLoop, dag, this,
      dag_actor_managers_[dag->Id()]->GetDagActorIds());
  }

  bool Stop() const {
    return env_->IsStopping();
  }

private:
  seastar::future<> RegisterDAG(const Dag *dag) {
    LOG(INFO) << "Register dag " << dag->Id();
    uint32_t dag_id = dag->Id();
    DagActorManager* manager = new DagActorManager(dag);
    dag_actor_managers_[dag_id] = manager;

    return seastar::parallel_for_each(
        boost::irange(0u, brane::local_shard_count()),
        [manager] (uint32_t i) mutable {
      brane::address shard_addr(i + brane::machine_info::sid_anchor());
      return brane::actor_client::request<brane::Boolean, RegisterMessage>(
        shard_addr, 0, RegisterMessage(manager),
        brane::message_type::REGISTER).discard_result();
    }).then([dag_id] {
      return brane::coordinator::get().global_barrier(
        "REGISTER_DAG_" + std::to_string(dag_id), false);
    });
  }

private:
  // dag id -> dag actor manager
  std::unordered_map<uint32_t, DagActorManager*> dag_actor_managers_;
  std::vector<std::thread> run_loops_;
};

void RunnerLoop(const Dag* dag, ActorDagScheduler *self,
                const std::vector<ActorIdType> *dag_actor_ids) {
  auto dispatcher = actor::NewTapeDispatcher(dag_actor_ids, dag->Root());
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
  return new actor::ActorDagScheduler(env);
}

}  // namespace graphlearn
