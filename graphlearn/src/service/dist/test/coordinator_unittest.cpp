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

#include "graphlearn/common/base/log.h"
#include "graphlearn/include/config.h"
#include "graphlearn/platform/env.h"
#include "graphlearn/service/dist/coordinator.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

class CoordinatorTest : public ::testing::Test {
public:
  CoordinatorTest() {
    InitGoogleLogging();
  }
  ~CoordinatorTest() {
    UninitGoogleLogging();
  }

protected:
  void SetUp() override {
    SetGlobalFlagTracker("./tracker");
    system("mkdir -p ./tracker");
  }

  void TearDown() override {
    system("rm -rf ./tracker");
  }
};

TEST_F(CoordinatorTest, StartReadyStop) {
  Coordinator* coord_0 = GetCoordinator(0, 2, Env::Default());
  Coordinator* coord_1 = GetCoordinator(1, 2, Env::Default());

  EXPECT_EQ(coord_0->IsMaster(), true);
  EXPECT_EQ(coord_1->IsMaster(), false);

  EXPECT_EQ(coord_0->IsStartup(), false);
  EXPECT_EQ(coord_1->IsStartup(), false);
  EXPECT_EQ(coord_0->IsReady(), false);
  EXPECT_EQ(coord_1->IsReady(), false);
  EXPECT_EQ(coord_0->IsStopped(), false);
  EXPECT_EQ(coord_1->IsStopped(), false);

  // 0 start
  Status s = coord_0->Start();
  EXPECT_TRUE(s.ok());
  // waiting for refresh
  sleep(2);

  // false, because of not all the servers have started
  EXPECT_EQ(coord_0->IsStartup(), false);
  EXPECT_EQ(coord_1->IsStartup(), false);
  EXPECT_EQ(coord_0->IsReady(), false);
  EXPECT_EQ(coord_1->IsReady(), false);
  EXPECT_EQ(coord_0->IsStopped(), false);
  EXPECT_EQ(coord_1->IsStopped(), false);
  // 1 start
  s = coord_1->Start();
  EXPECT_TRUE(s.ok());
  // waiting for refresh
  sleep(2);
  // start true, because of all the servers have started
  EXPECT_EQ(coord_0->IsStartup(), true);
  EXPECT_EQ(coord_1->IsStartup(), true);
  EXPECT_EQ(coord_0->IsReady(), false);
  EXPECT_EQ(coord_1->IsReady(), false);
  EXPECT_EQ(coord_0->IsStopped(), false);
  EXPECT_EQ(coord_1->IsStopped(), false);

  // 0 ready
  s = coord_0->Prepare();
  EXPECT_TRUE(s.ok());
  // waiting for refresh
  sleep(2);
  // false, because of not all the servers have been ready
  EXPECT_EQ(coord_0->IsStartup(), true);
  EXPECT_EQ(coord_1->IsStartup(), true);
  EXPECT_EQ(coord_0->IsReady(), false);
  EXPECT_EQ(coord_1->IsReady(), false);
  EXPECT_EQ(coord_0->IsStopped(), false);
  EXPECT_EQ(coord_1->IsStopped(), false);
  // 1 ready
  s = coord_1->Prepare();
  EXPECT_TRUE(s.ok());
  // waiting for refresh
  sleep(2);
  // ready true, because of all the servers have been ready
  EXPECT_EQ(coord_0->IsStartup(), true);
  EXPECT_EQ(coord_1->IsStartup(), true);
  EXPECT_EQ(coord_0->IsReady(), true);
  EXPECT_EQ(coord_1->IsReady(), true);
  EXPECT_EQ(coord_0->IsStopped(), false);
  EXPECT_EQ(coord_1->IsStopped(), false);

  // 0 stop
  s = coord_0->Stop(0, 4);
  EXPECT_TRUE(s.ok());
  s = coord_0->Stop(1, 4);
  EXPECT_TRUE(s.ok());
  // waiting for refresh
  sleep(2);
  // false, because of not all the servers have stopped
  EXPECT_EQ(coord_0->IsStartup(), true);
  EXPECT_EQ(coord_1->IsStartup(), true);
  EXPECT_EQ(coord_0->IsReady(), true);
  EXPECT_EQ(coord_1->IsReady(), true);
  EXPECT_EQ(coord_0->IsStopped(), false);
  EXPECT_EQ(coord_1->IsStopped(), false);
  // 1 stop
  s = coord_1->Stop(2, 4);
  EXPECT_TRUE(s.ok());
  s = coord_1->Stop(3, 4);
  EXPECT_TRUE(s.ok());
  // waiting for refresh
  sleep(2);
  // stop true, because of all the servers have stopped
  EXPECT_EQ(coord_0->IsStartup(), true);
  EXPECT_EQ(coord_1->IsStartup(), true);
  EXPECT_EQ(coord_0->IsReady(), true);
  EXPECT_EQ(coord_1->IsReady(), true);
  EXPECT_EQ(coord_0->IsStopped(), true);
  EXPECT_EQ(coord_1->IsStopped(), true);

  delete coord_0;
  delete coord_1;
}
