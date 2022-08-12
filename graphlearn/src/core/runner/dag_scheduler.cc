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

#include <utility>
#include "common/base/errors.h"
#include "common/base/log.h"
#include "core/dag/tape.h"
#include "core/runner/dag_node_runner.h"
#include "include/config.h"

namespace graphlearn {

class ThreadDagScheduler : public DagScheduler {
public:
  explicit ThreadDagScheduler(Env* env) : DagScheduler(env) {
    tp_ = env->IntraThreadPool();
    node_runner_ = new DagNodeRunner(env);
  }

  ~ThreadDagScheduler() {
    delete node_runner_;
  }

  void Run(const Dag* dag) override {
    tp_->AddTask(NewClosure(this, &ThreadDagScheduler::Start, dag));
  }

private:
  typedef std::function<void()> DoneCallback;

  void Start(const Dag* dag) {
    TapeStorePtr store = GetTapeStore(dag->Id());
    if (store == nullptr) {
      LOG(FATAL) << "Dag " << dag->Id() << " hasn't been registered.";
      return;
    }
    while (!Stop()) {
      /// The dag will be executed round by round in the background until
      /// the server stops. The results of each running round will be dumped
      /// to a Tape. The capacity of TapeStore is limited to the memory size.
      /// TapeStore can not generate a new tape until some old ones are
      /// consumed by clients.
      Tape* tape = store->New();
      KickOff(dag->Root(), tape);
      store->WaitAndPush(tape, [this](){
        return Stop();
      });
    }
  }

  bool Stop() {
    return env_->IsStopping();
  }

  void Submit(const DagNode* node, Tape* tape) {
    tp_->AddTask(
      NewClosure(this, &ThreadDagScheduler::KickOff, node, tape));
  }

  void KickOff(const DagNode* node, Tape* tape) {
    node_runner_->Run(node, tape);
    if (tape->IsFaked() || tape->IsReady()) {
      return;
    }

    DagNode* dag_node = const_cast<DagNode*>(node);
    dag_node->Send([this, tape](DagNode* dst){
      if (tape->IsReadyFor(dst)) {
        Submit(dst, tape);
      }
    });
  }

private:
  ThreadPool* tp_;
  DagNodeRunner* node_runner_;
};

DagScheduler* NewDefaultDagScheduler(Env* env) {
  return new ThreadDagScheduler(env);
}

#ifndef OPEN_ACTOR_ENGINE
DagScheduler* NewActorDagScheduler(Env* env) {
  USER_LOG("Hiactor is disabled! Using default dag scheduler.");
  return new ThreadDagScheduler(env);
}
#endif

DagScheduler::DagScheduler(Env* env)
    : env_(env) {
  optimizer_ = new Optimizer();
}

DagScheduler::~DagScheduler() {
  delete optimizer_;
}

void DagScheduler::Take(Env* env, const Dag* dag) {
  if (GLOBAL_FLAG(EnableActor) < 1) {
    static DagScheduler* scheduler = NewDefaultDagScheduler(env);
    scheduler->Run(dag);
  } else {
    static DagScheduler* scheduler = NewActorDagScheduler(env);
    scheduler->Run(dag);
  }
}

}  // namespace graphlearn
