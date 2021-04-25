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

#ifndef GRAPHLEARN_COMMON_THREADING_RUNNER_THREADPOOL_BASE_H_
#define GRAPHLEARN_COMMON_THREADING_RUNNER_THREADPOOL_BASE_H_

#include "graphlearn/common/base/closure.h"
#include "graphlearn/common/base/uncopyable.h"
#include "graphlearn/platform/protobuf.h"

namespace graphlearn {

class ThreadPoolBase : private Uncopyable {
public:
  typedef ::PB_NAMESPACE::Closure Callback;

  static const int kNotRunning = -1;
  static const int kQueueFull = -2;

  virtual ~ThreadPoolBase() { }

  // Startup enable the threadpool, and Shutdown turn it off. The
  // threadpool handles the tasks only when it's enabled.
  virtual bool Startup() = 0;
  virtual bool Shutdown() = 0;

  // Add task. The tasks add into threadpool is not cancelable, they will
  // be executed definitely, unless the threadpool is shutdown before the
  // tasks be executed.
  virtual int AddTask(Callback* task) = 0;

  // Wait until all the tasks added into the threadpool are executed. But
  // the client cannot assume the threadpool definitely are idle, unless
  // it sures that there isn't any more tasks added when calling WaitFor-
  // Idle().
  virtual void WaitForIdle() = 0;

  // Get pending task numbers.
  virtual int GetQueueLength() = 0;

  // Get live worker thread number in the threadpool.
  virtual int GetThreadNum() const = 0;

  // Get the thread numbers of the worker threads which is not idle.
  virtual int GetBusyThreadNum() const = 0;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_THREADING_RUNNER_THREADPOOL_BASE_H_
