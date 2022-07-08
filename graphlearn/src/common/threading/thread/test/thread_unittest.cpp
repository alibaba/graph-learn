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

#include "common/threading/thread/thread.h"

#include <semaphore.h>
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

static int sCount = 0;
static sem_t sCount_added;

void ThreadFunc() {
  ++sCount;
  sem_post(&sCount_added);
}

struct Foo {
  int count;
  sem_t count_added{};

  Foo() : count(0) {
    sem_init(&count_added, 0, 0);
  }
  ~Foo() {
    sem_destroy(&count_added);
  }

  void Bar() {
    ++count;
    sem_post(&count_added);
  }
};

TEST(ThreadTest, LaunchWithFunction) {
  sem_init(&sCount_added, 0, 0);
  Closure<void>* func = NewClosure(&ThreadFunc);
  ThreadHandle handle = CreateThread(func);
  sem_wait(&sCount_added);
  sem_destroy(&sCount_added);
  EXPECT_NE(0u, handle);
  EXPECT_EQ(1, sCount);
}

TEST(ThreadTest, LaunchWithMethod) {
  Foo foo;
  Closure<void>* func = NewClosure(&foo, &Foo::Bar);
  ThreadHandle handle = CreateThread(func);
  sem_wait(&foo.count_added);
  EXPECT_NE(0u, handle);
  EXPECT_EQ(1, foo.count);
}

