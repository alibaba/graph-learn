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

#ifndef GRAPHLEARN_CORE_RUNNER_DAG_SCHEDULER_H_
#define GRAPHLEARN_CORE_RUNNER_DAG_SCHEDULER_H_

#include <string>
#include "graphlearn/core/dag/optimizer.h"
#include "graphlearn/platform/env.h"

namespace graphlearn {

class DagScheduler {
public:
  /// Try to run the dag.
  static void Take(Env* env, const Dag* dag);

  virtual ~DagScheduler();

  /// A virtual method to run the dag, which is called by Take().
  virtual void Run(const Dag* dag) = 0;

protected:
  explicit DagScheduler(Env* env);

protected:
  Env*       env_;
  Optimizer* optimizer_;
};

DagScheduler* NewDefaultDagScheduler(Env* env);
DagScheduler* NewActorDagScheduler(Env* env);

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_RUNNER_DAG_SCHEDULER_H_
