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

#include "graphlearn/common/threading/lockfree/lockfree_queue.h"

#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

TEST(LockFreeQueueTest, Basic) {
  LockFreeQueue<int> queue;
  for (int k = 0; k < 16; ++k) {
    size_t ret = queue.Push(k);
    EXPECT_EQ(ret, (size_t)k + 1);
  }

  for (int k = 0; k < 16; ++k) {
    int i;
    EXPECT_EQ(queue.Pop(&i), true);
    EXPECT_EQ(i, k);
  }

  int p;
  EXPECT_EQ(queue.Pop(&p), false);
}
