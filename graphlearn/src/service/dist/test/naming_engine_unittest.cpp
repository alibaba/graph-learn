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
#include "graphlearn/service/dist/naming_engine.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

class NamingEngineTest : public ::testing::Test {
public:
  NamingEngineTest() {
    InitGoogleLogging();
  }
  ~NamingEngineTest() {
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

TEST_F(NamingEngineTest, CRUD) {
  NamingEngine* engine = NamingEngine::GetInstance();
  engine->SetCapacity(10);

  // init
  EXPECT_EQ(engine->Size(), 0);
  EXPECT_EQ(engine->Get(0), "");

  // update the first 5
  for (int32_t i = 0; i < 5; ++i) {
    Status s = engine->Update(i, std::to_string(i));
    EXPECT_TRUE(s.ok());
  }
  // waiting for refresh
  sleep(2);
  // check the first 5
  for (int32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(engine->Get(i), std::to_string(i));
  }
  // check the second 5, expect empty string
  for (int32_t i = 5; i < 10; ++i) {
    EXPECT_EQ(engine->Get(i), "");
  }
  // update the second 5
  for (int32_t i = 5; i < 10; ++i) {
    Status s = engine->Update(i, std::to_string(i));
    EXPECT_TRUE(s.ok());
  }
  // waiting for refresh
  sleep(2);
  // check the second 5
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(engine->Get(i), std::to_string(i));
  }

  // rewrite
  for (int32_t i = 0; i < 10; ++i) {
    Status s = engine->Update(i, std::to_string(i * 2));
    EXPECT_TRUE(s.ok());
  }
  // waiting for refresh
  sleep(2);
  // check
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(engine->Get(i), std::to_string(i * 2));
  }

  engine->Stop();
}
