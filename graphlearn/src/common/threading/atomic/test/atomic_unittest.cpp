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

#include "common/threading/atomic/atomic.h"

#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

struct Foo {
  int  value;
  Foo* next;
};

struct Bar {
  Foo *foo;

  Bar() : foo(0) { }
  void Add(int value) {
    Foo *old;
    Foo *f = new Foo();
    f->value = value;
    do {
      old = foo;
      f->next = old;
    } while (!AtomicCompareExchange(&foo, f, old));
  }

  void Sub(int *value) {
    Foo *old;
    Foo *f;
    do {
      old = foo;
      f = foo->next;
    } while (!AtomicCompareExchange(&foo, f, old));
    AtomicSub(value, old->value);
  }
};

TEST(AtomicTest, Increment) {
  int n = 0;
  int m = AtomicIncrement(&n);
  EXPECT_EQ(m, 1);
  EXPECT_EQ(n, 1);
  AtomicAdd(&n, 3);
  EXPECT_EQ(n, 4);
}

TEST(AtomicTest, ExchangeAdd) {
  int n = 5;
  EXPECT_EQ(AtomicExchangeAdd(&n, 4), 5);
  EXPECT_EQ(n, 9);
}

TEST(AtomicTest, SetGet) {
  int n = 0;
  AtomicSet(&n, 4);
  EXPECT_EQ(AtomicGet(&n), 4);
}
