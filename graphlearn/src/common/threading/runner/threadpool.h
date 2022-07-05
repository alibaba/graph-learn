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

#ifndef GRAPHLEARN_COMMON_THREADING_RUNNER_THREADPOOL_H_
#define GRAPHLEARN_COMMON_THREADING_RUNNER_THREADPOOL_H_

#include <string>
#include "common/base/closure.h"
#include "common/base/uncopyable.h"
#include "common/threading/runner/threadpool_base.h"

namespace graphlearn {

class ThreadPool : public ThreadPoolBase {
public:
  enum Type {
    eDynamicWorker = 0,
  };

public:
  explicit ThreadPool(int thread_num,
                      const std::string& name = "threadpool-default",
                      Type type = eDynamicWorker);

  ~ThreadPool() {
    delete impl_;
  }

  bool Startup() override {
    return impl_->Startup();
  }

  bool Shutdown() override {
    return impl_->Shutdown();
  }

  int AddTask(Callback* task) override {
    return impl_->AddTask(task);
  }

  void WaitForIdle() override {
    impl_->WaitForIdle();
  }

  int GetQueueLength() override {
    return impl_->GetQueueLength();
  }

  int GetThreadNum() const override {
    return impl_->GetThreadNum();
  }

  int GetBusyThreadNum() const override {
    return impl_->GetBusyThreadNum();
  }

private:
  ThreadPoolBase* impl_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_THREADING_RUNNER_THREADPOOL_H_
