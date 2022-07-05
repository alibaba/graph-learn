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

#ifndef GRAPHLEARN_COMMON_THREADING_SYNC_COND_H_
#define GRAPHLEARN_COMMON_THREADING_SYNC_COND_H_

#include <pthread.h>
#include <unistd.h>
#include <atomic>
#include "common/base/uncopyable.h"
#include "common/threading/sync/lock.h"

namespace graphlearn {

class ConditionVariable : private Uncopyable {
public:
  explicit ConditionVariable(MutexBase* lock);
  ~ConditionVariable();

  void Wait();
  bool TimedWait(int64_t delayInMs);
  void Signal();
  void Broadcast();

private:
  pthread_mutex_t* lock_;
  pthread_cond_t   cond_;
};

class SyncVariable {
public:
  explicit SyncVariable(int32_t th);
  ~SyncVariable();

  void Inc();
  void Wait();

private:
  std::atomic<int32_t> cond_;
  int32_t              threshold_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_THREADING_SYNC_COND_H_
