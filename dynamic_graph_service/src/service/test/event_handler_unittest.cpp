/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "service/test/test_helper.h"
#include "service/event_handler.h"

using namespace dgs;
using namespace std::chrono_literals;

class EventHandlerTest : public ::testing::Test {
public:
  EventHandlerTest() : helper_(4, 2) {}
  ~EventHandlerTest() override = default;

protected:
  void SetUp() override {
    InitGoogleLogging();
    FLAGS_alsologtostderr = true;
    Schema::GetInstance().Init();
    helper_.Initialize();
  }

  void TearDown() override {
    UninitGoogleLogging();
  }

protected:
  ServingTestHelper helper_;
};

TEST_F(EventHandlerTest, EventhandlerFunctionality) {
  helper_.InstallQuery();
  helper_.MakeSampleStore();

  const WorkerId worker_id = 0;

  RecordPollingManager poller_manager;
  AdaptiveRateLimiter rate_limiter(&poller_manager, 1);
  rate_limiter.Start();
  auto *event_handler = new EventHandler(worker_id, 10000, &rate_limiter);
  auto fut = seastar::alien::submit_to(
      *seastar::alien::internal::default_instance, 0,
      [event_handler] { return event_handler->Start(); });
  fut.wait();

  ServingTestHelper::SendRunQuery(2);

  auto fut2 = seastar::alien::submit_to(
      *seastar::alien::internal::default_instance, 0, [event_handler] {
    return seastar::sleep(1s).then([event_handler] () {
      return event_handler->Stop().then([event_handler] {
        delete event_handler;
      });
    });
  });
  fut2.wait();

  rate_limiter.Stop();
}
