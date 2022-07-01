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

#include "graphlearn/common/threading/lockfree/lockfree_stack.h"

#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

struct Foo {
  int mValue;
};

TEST(LockFreeStackTest, Basic) {
  const int count = 16;
  LockFreeStack<int> stack(16);
  EXPECT_EQ(stack.Size(), 0u);
  int bar;
  EXPECT_TRUE(!stack.Pop(&bar));

  for (int k = 0; k < count; ++k) {
    EXPECT_TRUE(stack.Push(k));
  }
  EXPECT_EQ(stack.Size(), (size_t)count);

  for (int k = 0; k < count; ++k) {
    EXPECT_TRUE(stack.Pop(&bar));
    EXPECT_EQ(bar, count - 1 - k);
  }
  EXPECT_EQ(stack.Size(), 0u);
}
