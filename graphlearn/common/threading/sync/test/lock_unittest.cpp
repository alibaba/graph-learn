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

#include "graphlearn/common/threading/sync/lock.h"

#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

TEST(LockTest, NullLock) {
  using LockType = NullLock;
  LockType lock;
  LockType::Locker locker(lock);
  LockType::Unlocker unlocker(lock);
}

TEST(LockTest, SpinLock) {
  using LockType = SpinLock;
  LockType lock;
  LockType::Locker locker(lock);
  LockType::Unlocker unlocker(lock);
}

TEST(LockTest, SimpleMutex) {
  using LockType = SimpleMutex;
  LockType lock;
  LockType::Locker locker(lock);
  LockType::Unlocker unlocker(lock);
}

TEST(LockTest, RestrictMutex) {
  using LockType = RestrictMutex;
  LockType lock;
  LockType::Locker locker(lock);
  LockType::Unlocker unlocker(lock);
}

TEST(LockTest, RecursiveMutex) {
  using LockType = RecursiveMutex;
  LockType lock;
  LockType::Locker locker1(lock);
  LockType::Locker locker2(lock);
  LockType::Unlocker unlocker(lock);
}

TEST(LockTest, AdaptiveMutex) {
  using LockType = AdaptiveMutex;
  LockType lock;
  LockType::Locker locker(lock);
  LockType::Unlocker unlocker(lock);
}

TEST(LockTest, RWLock) {
  using LockType = RWLock;
  LockType lock;
  {
    LockType::ReaderLocker locker(lock);
  }
  {
    LockType::WriterLocker locker(lock);
  }
}

TEST(LockTest, SpinRWLock) {
  using LockType = SpinRWLock;
  LockType lock;
  {
    LockType::ReaderLocker locker(lock);
  }
  {
    LockType::WriterLocker locker(lock);
  }
}

