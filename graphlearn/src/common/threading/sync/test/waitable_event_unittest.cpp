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

#include "common/threading/sync/waitable_event.h"

#include "gtest/gtest.h"
#include "common/base/closure.h"
#include "common/threading/thread/thread.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

namespace {

void Count(WaitableEvent* event, int64_t delayInMs, int64_t count) {
  while (count-- > 0) {
    if (!event[0].Wait(delayInMs)) {
      break;
    }
    event[1].Set();
  }
}

void StartWorkerThread(WaitableEvent* event, int64_t delayInMs, int64_t count) {
  Closure<void>* task = NewClosure(&Count, event, delayInMs, count);
  CreateThread(task);
}

}  // anonymous namespace

TEST(WaitableEventTest, DISABLED_Basic) {
  WaitableEvent event[2];
  int64_t count = 1024;

  StartWorkerThread(&event[0], -1, count);

  for (int k = 0; k < count; ++k) {
    event[0].Set();
    event[1].Wait();
  }
}

TEST(WaitableEventTest, DISABLED_WaitAndDtor) {
  for (int k = 0; k < 10000; ++k) {
    WaitableEvent event;
    Closure<void>* closure = NewClosure(&event, &WaitableEvent::Set);
    CreateThread(closure);
    event.Wait();
  }
}

