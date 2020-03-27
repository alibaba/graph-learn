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

#include "graphlearn/common/threading/runner/dynamic_worker_threadpool.h"

#include <cassert>
#include <set>
#include "gtest/gtest.h"
#include "graphlearn/common/threading/sync/lock.h"
#include "graphlearn/common/threading/this_thread.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

namespace {

struct DynamicWorkerThreadPoolEx : public DynamicWorkerThreadPool {
  DynamicWorkerThreadPoolEx(int thread_num, const std::string& name)
    : DynamicWorkerThreadPool(thread_num, name) { }

  void SetWorkerIdleWaitTimeBeforeTerminate(int ms) {
    return SetIdleThresholdInMs(ms);
  }
};

class Foo {
public:
  int Size() const {
    LockType::Locker locker(&mLock);
    return static_cast<int>(set_.size());
  }

  bool Exist(int k) const {
    LockType::Locker locker(&mLock);
    return set_.find(k) != set_.end();
  }
  void Insert(int k) {
    // the test task is too light, use sleep to ensure other working
    // threads involved.
    ThisThread::SleepInUs(1);
    LockType::Locker locker(&mLock);
    if (set_.find(k) != set_.end()) {
      assert(false);
    }
    set_.insert(k);
  }

  void Remove(int k) {
    LockType::Locker locker(&mLock);
    if (set_.find(k) == set_.end()) {
      assert(false);
    }
    set_.erase(k);
  }

private:
  using LockType = SpinLock;
  mutable LockType mLock;
  std::set<int> set_;
};

class Bar {
public:
  Bar() : value_(0) { }

  void Add(int k) {
    AtomicAdd(&value_, k);
    // the test task is too light, use sleep to ensure other working
    // threads involved.
    ThisThread::SleepInUs(10);
  }

  void Dummy() { }

  int Sum() const {
    return AtomicGet(&value_);
  }

private:
  int value_;
};

TEST(DynamicWorkerThreadPoolTest, StartAndShutdown) {
  DynamicWorkerThreadPoolEx threadpool(2, "testing");
  threadpool.Startup();
  threadpool.Shutdown();
}

TEST(DynamicWorkerThreadPoolTest, Set) {
  DynamicWorkerThreadPoolEx threadpool(5, "testing");
  Foo foo;
  threadpool.Startup();

  int count = 1000;
  for (int k = 0; k < count; ++k) {
    DynamicWorkerThreadPool::Callback *task =
        NewClosure(&foo, &Foo::Insert, k);
    threadpool.AddTask(task);
  }

  threadpool.WaitForIdle();
  EXPECT_EQ(count, foo.Size());

  for (int k = 0; k < count; ++k) {
    EXPECT_TRUE(foo.Exist(k));
  }

  for (int k = 0; k < count; ++k) {
    DynamicWorkerThreadPool::Callback *task =
        NewClosure(&foo, &Foo::Remove, k);
    threadpool.AddTask(task);
  }

  threadpool.WaitForIdle();
  EXPECT_EQ(0, foo.Size());

  threadpool.Shutdown();
}

TEST(DynamicWorkerThreadPoolTest, Sum) {
  DynamicWorkerThreadPoolEx threadpool(5, "testing");
  Bar bar;
  threadpool.Startup();

  int count = 1000;
  int sum = 0;
  for (int k = 0; k < count; ++k) {
    DynamicWorkerThreadPool::Callback *task =
        NewClosure(&bar, &Bar::Add, k);
    threadpool.AddTask(task);
    sum += k;
  }

  threadpool.WaitForIdle();
  EXPECT_EQ(5, threadpool.GetThreadNum());
  EXPECT_EQ(0, threadpool.GetBusyThreadNum());
  EXPECT_EQ(0, threadpool.GetQueueLength());

  threadpool.Shutdown();
}

TEST(DynamicWorkerThreadPoolTest, IdleWorker) {
  DynamicWorkerThreadPoolEx threadpool(5, "testing");
  threadpool.SetWorkerIdleWaitTimeBeforeTerminate(30);
  Bar bar;
  threadpool.Startup();

  int count = 1000;
  int sum = 0;
  for (int k = 0; k < count; ++k) {
    DynamicWorkerThreadPool::Callback *task =
        NewClosure(&bar, &Bar::Add, k);
    threadpool.AddTask(task);
    sum += k;
  }

  threadpool.WaitForIdle();
  EXPECT_EQ(5, threadpool.GetThreadNum());
  EXPECT_EQ(0, threadpool.GetBusyThreadNum());
  EXPECT_EQ(0, threadpool.GetQueueLength());

  // Ensure the idle worker threads exit.
  ThisThread::SleepInMs(60);

  EXPECT_EQ(1, threadpool.GetThreadNum());

  for (int k = 0; k < count; ++k) {
    DynamicWorkerThreadPool::Callback *task =
        NewClosure(&bar, &Bar::Add, k);
    threadpool.AddTask(task);
    sum += k;
  }

  threadpool.WaitForIdle();
  EXPECT_EQ(5, threadpool.GetThreadNum());
  EXPECT_EQ(0, threadpool.GetBusyThreadNum());
  EXPECT_EQ(0, threadpool.GetQueueLength());

  threadpool.Shutdown();
}

}  // anonymous namespace

