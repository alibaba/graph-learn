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

#include "common/base/log.h"
#include "include/config.h"
#include "platform/env.h"
#include "service/dist/channel_manager.h"
#include "service/dist/naming_engine.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

class ChannelManagerTest : public ::testing::Test {
public:
  ChannelManagerTest() {
    InitGoogleLogging();
  }
  ~ChannelManagerTest() {
    UninitGoogleLogging();
  }

protected:
  void SetUp() override {
    SetGlobalFlagTracker("./tracker");
    SetGlobalFlagServerCount(2);
    SetGlobalFlagClientCount(4);
    SetGlobalFlagRetryTimes(1);
    EXPECT_EQ(::system("mkdir -p ./tracker"), 0);
  }

  void TearDown() override {
    EXPECT_EQ(::system("rm -rf ./tracker"), 0);
  }
};

TEST_F(ChannelManagerTest, ConnectSelect) {
  ChannelManager* manager = ChannelManager::GetInstance();
  manager->SetCapacity(GLOBAL_FLAG(ServerCount));

  // all the channel is broken, because the naming engine not start
  SetGlobalFlagClientId(0);
  auto channel = manager->ConnectTo(0);
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(channel->IsBroken());
  channel = manager->AutoSelect();
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(channel->IsBroken());

  SetGlobalFlagClientId(1);
  channel = manager->ConnectTo(0);
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(channel->IsBroken());
  channel = manager->AutoSelect();
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(channel->IsBroken());

  SetGlobalFlagClientId(2);
  channel = manager->ConnectTo(1);
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(channel->IsBroken());
  channel = manager->AutoSelect();
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(channel->IsBroken());

  SetGlobalFlagClientId(3);
  channel = manager->ConnectTo(1);
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(channel->IsBroken());
  channel = manager->AutoSelect();
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(channel->IsBroken());

  NamingEngine* engine = NamingEngine::GetInstance();
  engine->Update(0, "127.0.0.1:5678");
  engine->Update(1, "127.0.0.1:6789");

  // waiting for refresh
  sleep(3);

  // now the channel is not broken
  SetGlobalFlagClientId(0);
  channel = manager->ConnectTo(0);
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(!channel->IsBroken());
  channel = manager->AutoSelect();
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(!channel->IsBroken());

  SetGlobalFlagClientId(1);
  channel = manager->ConnectTo(0);
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(!channel->IsBroken());
  channel = manager->AutoSelect();
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(!channel->IsBroken());

  SetGlobalFlagClientId(2);
  channel = manager->ConnectTo(1);
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(!channel->IsBroken());
  channel = manager->AutoSelect();
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(!channel->IsBroken());

  SetGlobalFlagClientId(3);
  channel = manager->ConnectTo(1);
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(!channel->IsBroken());
  channel = manager->AutoSelect();
  EXPECT_TRUE(channel != nullptr);
  EXPECT_TRUE(!channel->IsBroken());

  manager->Stop();
  engine->Stop();
}
