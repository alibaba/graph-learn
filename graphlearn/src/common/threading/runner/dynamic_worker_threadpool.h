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

#ifndef GRAPHLEARN_COMMON_THREADING_RUNNER_DYNAMIC_WORKER_THREADPOOL_H_
#define GRAPHLEARN_COMMON_THREADING_RUNNER_DYNAMIC_WORKER_THREADPOOL_H_

#include <string>
#include "common/threading/atomic/atomic.h"
#include "common/threading/lockfree/lockfree_queue.h"
#include "common/threading/lockfree/lockfree_stack.h"
#include "common/threading/runner/threadpool_base.h"
#include "common/threading/sync/lock.h"
#include "common/threading/sync/waitable_event.h"
#include "platform/protobuf.h"

namespace graphlearn {

//   This Threadpool is used to run instances of graphlearn::Closure, which is
//   compatible with PB_NAMESPACE::Closure.  The number of worker threads
//   will adjusts dynamically.  When adding tasks, it may add worker thread
//   if 1) all the worker threads are processing tasks and, 2) the current
//   worker threads number is less than thread_num limit of this thread pool.
//   If the thread pool is idle for 79s, the worker threads will terminate
//   to save system resources, and keeps at least one work thread alive.

class DynamicWorkerThreadPool : public ThreadPoolBase {
public:
  explicit DynamicWorkerThreadPool(
      int thread_num,
      const std::string& name = "threadpool-default");

  ~DynamicWorkerThreadPool();

  bool Startup() override;
  bool Shutdown() override;

  int AddTask(Callback* callback) override;
  void WaitForIdle() override;

  int GetQueueLength() override;
  int GetThreadNum() const override;
  int GetBusyThreadNum() const override;

protected:
  void SetIdleThresholdInMs(int ms);

private:
  struct Task {
    Callback* callback_;
  };

  struct ThreadInfo {
    WaitableEvent event_;
  };

  bool IsRunning() const;
  void AddWorkerThread();
  void WorkerRoutine();

  // Utility methods for maintaining worker thread.
  bool WaitForNotify(ThreadInfo* info);
  void PushIdleThread(ThreadInfo* info);
  bool PopIdleThread(ThreadInfo** info);
  bool RemoveIdleThread(ThreadInfo* info);

  void ExecuteOneTask(Task* task);
  void AtWorkerExit();
  // Get a Task instance from cached free-list, or allocate a new one.
  bool AcquireTask(Task** task);
  // Store the Task instance into cached free-list, or release the memory
  // if there are enough instances cached already.
  void ReleaseTask(Task* task);

private:
  // the time interval before idle thread to exit, in millisecond.
  static const int kIdleThresholdInMs;
  // the maximum thread number allowed in a single thread pool
  static const int kMaxThreadNum;
  // the minimum thread number in a single thread pool
  static const int kMinThreadNum;
  // the maximum free tasks in cache list allowed
  static const int kMaxTasksInCache;

  const std::string name_;
  WaitableEvent event_for_all_workers_exit_;

  const int max_thead_num_;
  int thread_num_;
  int idle_thread_num_;
  int idle_threshold_in_ms_;

  LockFreeStack<ThreadInfo*> idle_thread;

  // the read write lock guards the start and stop of this thread pool.
  typedef SpinRWLock RWLockType;
  RWLockType rw_lock_;
  bool started_;
  bool stopped_;

  LockFreeQueue<Task*> tasks_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_THREADING_RUNNER_DYNAMIC_WORKER_THREADPOOL_H_
