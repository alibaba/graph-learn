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

#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

static int sCount = 0;

void ThreadFunc() {
  ++sCount;
}

struct Foo {
  int count_;

  Foo() : count_(0) { }

  void Bar() {
    ++count_;
  }
};

TEST(ThreadTest, LaunchWithFunction) {
    Closure<void>* func = NewClosure(&ThreadFunc);
    ThreadHandle handle = CreateThread(func);
    ::usleep(1000000);
    EXPECT_NE(0u, handle);
    EXPECT_EQ(1, sCount);
}

TEST(ThreadTest, LaunchWithMethod) {
    Foo foo;
    Closure<void>* func = NewClosure(&foo, &Foo::Bar);
    ThreadHandle handle = CreateThread(func);
    ::usleep(1000000);
    EXPECT_NE(0u, handle);
    EXPECT_EQ(1, foo.count_);
}

