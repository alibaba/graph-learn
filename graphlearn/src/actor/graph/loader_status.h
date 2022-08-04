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

#ifndef GRAPHLEARN_ACTOR_GRAPH_LOADER_STATUS_H_
#define GRAPHLEARN_ACTOR_GRAPH_LOADER_STATUS_H_

#include <semaphore.h>

namespace graphlearn {
namespace act {

class DataLoaderStatus {
  static DataLoaderStatus* Get() {
    static DataLoaderStatus inst;
    return &inst;
  }

  DataLoaderStatus() {
    sem_init(&finished_, 0, 0);
  }

  ~DataLoaderStatus() {
    sem_destroy(&finished_);
  }

  void WaitUntilFinished() {
    sem_wait(&finished_);
  }

  void NotifyFinished() {
    sem_post(&finished_);
  }

private:
  sem_t finished_;

  friend class ControlActor;
  friend class ShardedGraphStore;
};

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_GRAPH_LOADER_STATUS_H_
