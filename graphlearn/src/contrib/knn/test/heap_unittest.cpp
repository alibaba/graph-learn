// Copyright (c) 2019, Alibaba Inc.
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

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "graphlearn/contrib/knn/heap.h"
#include "gtest/gtest.h"

namespace graphlearn {
namespace op {

template<class T>
void shuffle(T a[], int n) {
  srand(time(NULL));

  for (int i = n - 1; i > 0; --i) {
    int index = rand() % i; //NOLINT [runtime/threadsafe_fn]
    std::swap(a[i], a[index]);
  }
}


class HeapTest : public ::testing::Test {
protected:
  void SetUp() override {
  }

  void TearDown() override {
  }
};

TEST_F(HeapTest, MaxHeap) {
  Heap<int64_t> heap(10);

  int64_t ids[100];
  for (int i = 0; i < 100; ++i) {
    ids[i] = i;
  }

  shuffle(ids, 100);

  float distances[100];
  for (int i = 0; i < 100; ++i) {
    distances[i] = static_cast<float>(ids[i]);
  }

  ASSERT_TRUE(heap.Empty());
  EXPECT_EQ(heap.Size(), 0);

  // the heap will hold from 0 to 9
  for (int i = 0; i < 100; ++i) {
    heap.Push(distances[i], ids[i]);

    ASSERT_TRUE(!heap.Empty());

    if (i < 10) {
      EXPECT_EQ(heap.Size(), i + 1);
    } else {
      EXPECT_EQ(heap.Size(), 10);
    }
  }

  // the heap will pop from 9 to 0
  for (int i = 10; i > 0; --i) {
    float value = 0.0;
    int64_t id = -1;
    bool exist = heap.Pop(&value, &id);
    ASSERT_TRUE(exist);
    EXPECT_EQ(heap.Size(), i - 1);
    EXPECT_EQ(id, i - 1);
    EXPECT_FLOAT_EQ(value, static_cast<float>(i - 1));
  }

  ASSERT_TRUE(heap.Empty());

  for (int i = 0; i < 100; ++i) {
    heap.Push(distances[i], ids[i]);
  }
  ASSERT_TRUE(!heap.Empty());
  heap.Clear();
  ASSERT_TRUE(heap.Empty());
}

TEST_F(HeapTest, MinHeap) {
  Heap<int64_t, MaxCompare> heap(10);

  int64_t ids[100];
  for (int i = 0; i < 100; ++i) {
    ids[i] = i;
  }

  shuffle(ids, 100);

  float distances[100];
  for (int i = 0; i < 100; ++i) {
    distances[i] = static_cast<float>(ids[i]);
  }

  ASSERT_TRUE(heap.Empty());
  EXPECT_EQ(heap.Size(), 0);

  // the heap will hold from 90 to 99
  for (int i = 0; i < 100; ++i) {
    heap.Push(distances[i], ids[i]);

    ASSERT_TRUE(!heap.Empty());

    if (i < 10) {
      EXPECT_EQ(heap.Size(), i + 1);
    } else {
      EXPECT_EQ(heap.Size(), 10);
    }
  }

  // the heap will pop from 90 to 99
  for (int i = 10; i > 0; --i) {
    float value = 0.0;
    int64_t id = -1;
    bool exist = heap.Pop(&value, &id);
    ASSERT_TRUE(exist);
    EXPECT_EQ(heap.Size(), i - 1);
    EXPECT_EQ(id, 100 - i);
    EXPECT_FLOAT_EQ(value, static_cast<float>(id));
  }

  ASSERT_TRUE(heap.Empty());

  for (int i = 0; i < 100; ++i) {
    heap.Push(distances[i], ids[i]);
  }
  ASSERT_TRUE(!heap.Empty());
  heap.Clear();
  ASSERT_TRUE(heap.Empty());
}

}  // namespace op
}  // namespace graphlearn
