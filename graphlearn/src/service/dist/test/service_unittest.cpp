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

#include <thread>
#include "common/base/log.h"
#include "include/config.h"
#include "platform/env.h"
#include "service/dist/channel_manager.h"
#include "service/dist/coordinator.h"
#include "service/dist/grpc_service.h"
#include "service/dist/naming_engine.h"
#include "service/dist/service.h"
#include "service/executor.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

class DistributeServiceTest : public ::testing::Test {
public:
  DistributeServiceTest() {
    InitGoogleLogging();
  }
  ~DistributeServiceTest() {
    UninitGoogleLogging();
  }

protected:
  void SetUp() override {
    SetGlobalFlagTracker("./tracker");
    EXPECT_EQ(::system("mkdir -p ./tracker"), 0);
  }

  void TearDown() override {
    EXPECT_EQ(::system("rm -rf ./tracker"), 0);
  }
};

void RequestToStop(Coordinator* coord) {
  coord->Stop(0, 1);
}

TEST_F(DistributeServiceTest, StartInitStopWithoutHosts) {
  Env* env = Env::Default();
  Executor* executor = nullptr;  // not handle request, just set null
  int32_t server_id = 0;
  int32_t server_count = 1;

  Coordinator* coordinator = GetCoordinator(server_id, server_count, env);

  DistributeService* service = new DistributeService(
    server_id, server_count, "", env, executor, coordinator);
  Status s = service->Start();
  EXPECT_TRUE(s.ok());
  s = service->Init();
  EXPECT_TRUE(s.ok());

  // here we use a thread to mock client has request to stop
  Coordinator* coord = service->GetCoordinator();
  std::thread* t = new std::thread(&RequestToStop, coord);

  s = service->Stop();
  EXPECT_TRUE(s.ok());

  t->join();
  delete t;
  delete service;
  delete coordinator;
}
