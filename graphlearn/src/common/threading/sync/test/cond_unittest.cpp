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

#include "graphlearn/common/threading/sync/cond.h"

#include "gtest/gtest.h"
#include "graphlearn/common/threading/atomic/atomic.h"
#include "graphlearn/common/threading/this_thread.h"
#include "graphlearn/common/threading/thread/thread.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

namespace {

void Produce(SimpleMutex* mutex,
             ConditionVariable* cond,
             int delayInMs,
             bool broadcast) {
  ThisThread::SleepInMs(delayInMs);
  SimpleMutex::Locker _(*mutex);
  if (broadcast) {
    cond->Broadcast();
  } else {
    cond->Signal();
  }
}

void Consume(SimpleMutex* mutex,
             ConditionVariable* cond,
             int* count) {
  {
    SimpleMutex::Locker _(*mutex);
    cond->Wait();
  }
  AtomicIncrement(count);
}

void StartProducer(SimpleMutex* mutex,
                   ConditionVariable* cond,
                   int delayInMs,
                   bool broadcast) {
  Closure<void>* task =
    NewClosure(&Produce, mutex, cond, delayInMs, broadcast);
  CreateThread(task);
}

void StartConsumer(SimpleMutex* mutex,
                   ConditionVariable* cond,
                   int* count) {
  Closure<void>* task =
    NewClosure(&Consume, mutex, cond, count);
  CreateThread(task);
}

}  // anonymous namespace

TEST(ConditionVariableTest, Basic) {
  SimpleMutex mutex;
  ConditionVariable cond(&mutex);

  StartProducer(&mutex, &cond, 10, false);
  EXPECT_TRUE(cond.TimedWait(200));
}

TEST(ConditionVariableTest, Timeout) {
  SimpleMutex mutex;
  ConditionVariable cond(&mutex);

  StartProducer(&mutex, &cond, 50, false);
  EXPECT_TRUE(!cond.TimedWait(5));
}

TEST(ConditionVariableTest, DISABLED_Broadcast) {
  SimpleMutex mutex;
  ConditionVariable cond(&mutex);
  int count = 0;

  StartConsumer(&mutex, &cond, &count);
  StartConsumer(&mutex, &cond, &count);
  StartProducer(&mutex, &cond, 50, true);
  cond.Wait();
  EXPECT_EQ(count, 2);
}

