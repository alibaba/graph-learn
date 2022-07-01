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

#include <thread>  // NOLINT
#include <vector>
#include "graphlearn/service/local/event_queue.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT

class EventQueueTest : public ::testing::Test {
protected:
  void SetUp() override {
  }

  void TearDown() override {
  }
};

TEST_F(EventQueueTest, PushPopInSingleThread) {
  int32_t capacity = 100;
  EventQueue<int32_t> q(capacity);

  for (int32_t i = 0; i < capacity; ++i) {
    EXPECT_TRUE(q.Push(i));
  }

  int32_t value = 0;
  for (int32_t i = 0; i < capacity; ++i) {
    EXPECT_TRUE(q.Pop(&value));
    EXPECT_EQ(value, i);
  }

  q.Cancel();
  EXPECT_TRUE(!q.Push(0));
  EXPECT_TRUE(!q.Pop(&value));
}

void DoPush(EventQueue<int32_t>* q, int32_t start, int32_t count) {
  for (int32_t i = 0; i < count; ++i) {
    EXPECT_TRUE(q->Push(i + start));
  }
}

TEST_F(EventQueueTest, PushPopInMultiThread) {
  int32_t capacity = 100;
  EventQueue<int32_t> q(capacity);

  std::thread* t1 = new std::thread(&DoPush, &q, 0, 50);
  std::thread* t2 = new std::thread(&DoPush, &q, 50, 50);
  t1->join();
  t2->join();
  delete t1;
  delete t2;

  int32_t value = 0;
  std::vector<int32_t> values(capacity);
  for (int32_t i = 0; i < capacity; ++i) {
    EXPECT_TRUE(q.Pop(&value));
    values[value] = value;
  }

  for (int32_t i = 0; i < capacity; ++i) {
    EXPECT_EQ(values[i], i);
  }

  q.Cancel();
  EXPECT_TRUE(!q.Push(0));
  EXPECT_TRUE(!q.Pop(&value));
}
