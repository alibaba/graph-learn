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

#include <unistd.h>
#include <memory>
#include <thread>  // NOLINT [build/c++11]
#include "gtest/gtest.h"
#include "graphlearn/common/base/errors.h"
#include "graphlearn/core/rpc/base.h"

using namespace graphlearn::error;  // NOLINT [build/namespaces]
using namespace graphlearn::rpc;  // NOLINT [build/namespaces]

TEST(StatusWrapperTest, Normal) {
  StatusWrapper s1, s2;
  auto p_thread = std::thread([&] {
    ::usleep(10 * 1000);
    s1.Signal();
    s2.Signal(NotFound("not found."));
  });
  s1.Wait();
  s2.Wait();
  EXPECT_TRUE(s1.s.ok());
  EXPECT_TRUE(IsNotFound(s2.s));
  p_thread.join();
}

TEST(StatusWrapperTest, Timeout) {
  auto s1 = std::make_shared<StatusWrapper>();
  std::weak_ptr<StatusWrapper> weak(s1);
  auto p_thread = std::thread([&] {
    ::usleep(1000 * 1000);
    auto status = weak.lock();
    if (status) {
      status->Signal();
      }
  });
  s1->Wait(100);
  EXPECT_TRUE(IsCancelled(s1->s));
  p_thread.join();
}

