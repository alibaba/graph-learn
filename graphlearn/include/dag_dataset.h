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

#ifndef GRAPHLEARN_INCLUDE_DAG_DATASET_H_
#define GRAPHLEARN_INCLUDE_DAG_DATASET_H_

#include <mutex>  // NOLINT [build/c++11]
#include <semaphore.h>
#include "graphlearn/include/client.h"
#include "graphlearn/include/dag_request.h"
#include "graphlearn/common/threading/runner/threadpool.h"
#include "graphlearn/common/threading/sync/semaphore_shim.h"

namespace graphlearn {

class Dataset {
public:
  Dataset(Client* client, int32_t dag_id);
  ~Dataset();

  void Close();

  GetDagValuesResponse* Next(int32_t epoch);

private:
  void PrefetchAsync();
  void PrefetchFn();

private:
  Client* client_;
  int32_t dag_id_;
  int32_t cap_;
  int32_t cursor_;

#if __APPLE__
  std::vector<macos_sem_t> occupied_;
#else
  std::vector<sem_t> occupied_;
#endif
  std::atomic<int32_t> head_;
  std::unique_ptr<ThreadPool> tp_;
  std::vector<GetDagValuesResponse*> buffer_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_DAG_DATASET_H_
