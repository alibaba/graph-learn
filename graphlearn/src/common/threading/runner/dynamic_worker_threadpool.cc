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

#include "common/threading/runner/dynamic_worker_threadpool.h"

#include <unistd.h>
#include <stack>
#include "common/threading/this_thread.h"
#include "common/threading/thread/thread.h"

namespace graphlearn {

const int DynamicWorkerThreadPool::kIdleThresholdInMs = 79000;  // 79 seconds
const int DynamicWorkerThreadPool::kMaxThreadNum = 32;
const int DynamicWorkerThreadPool::kMinThreadNum = 1;

DynamicWorkerThreadPool::DynamicWorkerThreadPool(
  int thread_num, const std::string& name)
    : name_(name),
      max_thead_num_(std::min(thread_num, kMaxThreadNum)),
      thread_num_(0),
      idle_thread_num_(0),
      idle_threshold_in_ms_(kIdleThresholdInMs),
      idle_thread(thread_num),
      started_(false),
      stopped_(false) {
}

DynamicWorkerThreadPool::~DynamicWorkerThreadPool() {
  Shutdown();

  Task *task = nullptr;
  while (tasks_.Pop(&task)) {
    delete task->callback_;
    delete task;
  }
}

bool DynamicWorkerThreadPool::Startup() {
  RWLockType::WriterLocker locker(&rw_lock_);
  if (started_) {
    return false;
  }
  started_ = true;
  return true;
}

bool DynamicWorkerThreadPool::Shutdown() {
  {
    RWLockType::WriterLocker locker(&rw_lock_);
    if (!started_ || stopped_) {
      return false;
    }
    stopped_ = true;
  }
  {
    // If there isn't any active thread, it's dangrous, we need an active
    // thread at least, help to signal event_for_all_workers_exit_
    RWLockType::ReaderLocker locker(&rw_lock_);
    if (thread_num_ == 0) {
      AddWorkerThread();
    }
  }
  while (true) {
    ThreadInfo *info = nullptr;
    // Wake up all the idle worker threads, they will exit automatically.
    while (PopIdleThread(&info)) {
      info->event_.Set();
    }
    if (AtomicGet(&thread_num_) == 0) {
      break;
    }
    ThisThread::SleepInUs(1000);
  }
  // Wait for all the worker threads finish.
  event_for_all_workers_exit_.Wait();
  return true;
}

int DynamicWorkerThreadPool::AddTask(Callback* callback) {
  RWLockType::ReaderLocker locker(&rw_lock_);
  if (!IsRunning()) {
      return kNotRunning;
  }

  Task* task = nullptr;
  if (!AcquireTask(&task)) {
    return kQueueFull;
  }

  task->callback_ = callback;

  size_t pendingCount = tasks_.Push(task);

  // If there are any idle thread, wake it up now.
  ThreadInfo* info = nullptr;
  if (PopIdleThread(&info)) {
    info->event_.Set();
  }

  // Add more working thread if: 1) there isn't any more idle thread, and
  // 2) the thread number is less than the maximum allowed.
  if (info == nullptr && thread_num_ < max_thead_num_) {
    AddWorkerThread();
  }

  return std::max(static_cast<int>(pendingCount), 1);
}

void DynamicWorkerThreadPool::WaitForIdle() {
  while (true) {
    {
      RWLockType::ReaderLocker lockerer(&rw_lock_);
      if (tasks_.Empty() && thread_num_ == idle_thread_num_) {
        break;
      }
    }
    ThisThread::SleepInUs(1000);
  }
}

int DynamicWorkerThreadPool::GetQueueLength() {
  return static_cast<int>(tasks_.Size());
}

int DynamicWorkerThreadPool::GetThreadNum() const {
  return thread_num_;
}

int DynamicWorkerThreadPool::GetBusyThreadNum() const {
  return thread_num_ - AtomicGet(&idle_thread_num_);
}

void DynamicWorkerThreadPool::SetIdleThresholdInMs(int ms) {
  idle_threshold_in_ms_ = ms;
}

bool DynamicWorkerThreadPool::IsRunning() const {
  return started_ && !stopped_;
}

void DynamicWorkerThreadPool::AddWorkerThread() {
  if (AtomicIncrement(&thread_num_) > max_thead_num_) {
    AtomicDecrement(&thread_num_);
    return;
  }
  Closure<void>* func =
      NewClosure(this, &DynamicWorkerThreadPool::WorkerRoutine);
  CreateThread(func, nullptr, name_.c_str());
}

void DynamicWorkerThreadPool::WorkerRoutine() {
  ThreadInfo info;
  while (IsRunning()) {
    Task *task = nullptr;
    if (tasks_.Pop(&task)) {
      // execute it
      ExecuteOneTask(task);
      ReleaseTask(task);
    } else {
      AtomicIncrement(&idle_thread_num_);
      // Continue the loop when 1) the condition achieved, or
      // 2) the total thread count not more than threshold.
      if (WaitForNotify(&info)) {
        AtomicDecrement(&idle_thread_num_);
      } else {
        // Ensure to remove this thread from idle thread stack to
        // keep the consistency of information.
        // Be care the timing gap between the WaitForNotify and
        // RemoveIdleThread. There may be other thread has removed this
        // ThreadInfo from the top of stack. In this situation, the
        // ThreadInfo::event_ is set, and then its info object is not
        // on the stack.
        while (true) {
          if (info.event_.Wait(0) || RemoveIdleThread(&info)) {
            break;
          }
          ThisThread::Yield();
        }

        // In either case, if the control of a worker thread reaches
        // here, it means that its ThreadInfo object has been removed.
        AtomicDecrement(&idle_thread_num_);

        if (tasks_.Empty() && AtomicGet(&thread_num_) > kMinThreadNum) {
          break;
        }
      }
    }
  }

  // When terminating the threadpool, execute all the pending tasks.
  Task* task = nullptr;
  while (tasks_.Pop(&task)) {
    // execute it
    ExecuteOneTask(task);
    ReleaseTask(task);
  }

  AtWorkerExit();
}

bool DynamicWorkerThreadPool::WaitForNotify(ThreadInfo* info) {
  PushIdleThread(info);

  // If there is a new task coming, remove itself from the idle list
  // right now. there is performance punishment!
  // If there is another idle thread on the stack top now, wake it up.
  ThreadInfo *pinfo = nullptr;
  if (!tasks_.Empty() && PopIdleThread(&pinfo)) {
    if (pinfo == info) {
      return true;
    } else {
      pinfo->event_.Set();
    }
  }

  // If it's sure that there isn't pending task, go to wait for notify.
  if (info->event_.Wait(idle_threshold_in_ms_) == 1) {
    // Wake up now because get a notification.
    return true;
  }

  return false;
}

void DynamicWorkerThreadPool::PushIdleThread(ThreadInfo* info) {
  if (!idle_thread.Push(info)) {
    ::abort();
  }
}

bool DynamicWorkerThreadPool::PopIdleThread(ThreadInfo** info) {
  return idle_thread.Pop(info);
}

bool DynamicWorkerThreadPool::RemoveIdleThread(ThreadInfo *info) {
  std::stack<ThreadInfo*> threads;
  ThreadInfo *that = nullptr;
  bool found = false;
  while (PopIdleThread(&that)) {
    if (that == info) {
      found = true;
      break;
    }
  threads.push(that);
  }
  while (!threads.empty()) {
    that = threads.top();
    threads.pop();
    PushIdleThread(that);
  }

  return found;
}

void DynamicWorkerThreadPool::ExecuteOneTask(Task* task) {
  // If user task throws a exception, we cannot do anything, just ignore it,
  // and let the process aborts.  It's the user's responsibility to handle
  // all the exceptions.
  task->callback_->Run();
}

void DynamicWorkerThreadPool::AtWorkerExit() {
  bool notifyAllWorkersExit = false;
  {
    RWLockType::ReaderLocker locker(&rw_lock_);
    // worker thread is going to exit now
    AtomicDecrement(&thread_num_);
    // when existing, set the signal when the last thread finishes
    if (!IsRunning() && thread_num_ == 0) {
      notifyAllWorkersExit = true;
    }
  }
  // Do not notify in lock, because the object may be deleted at once
  // even before this thread release the lock.
  if (notifyAllWorkersExit) {
    event_for_all_workers_exit_.Set();
  }
}

bool DynamicWorkerThreadPool::AcquireTask(Task** task) {
  *task = new Task;
  return true;
}

void DynamicWorkerThreadPool::ReleaseTask(Task* task) {
  task->callback_ = nullptr;
  delete task;
}

}  // namespace graphlearn
