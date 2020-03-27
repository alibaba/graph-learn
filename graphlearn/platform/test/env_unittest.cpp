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
#include "graphlearn/platform/env.h"
#include "gtest/gtest.h"

using namespace graphlearn;  //NOLINT [build/namespaces]

class EnvTest : public ::testing::Test {
public:
  EnvTest() {
    InitGoogleLogging();
  }
  ~EnvTest() {
    UninitGoogleLogging();
  }
protected:
  void SetUp() override {
    env_ = Env::Default();
  }

  void TearDown() override {
  }

protected:
  Env* env_;
};

TEST_F(EnvTest, Singleton) {
  Env* env = Env::Default();
  EXPECT_EQ(env, env_);
}

TEST_F(EnvTest, FileSystemSupported) {
  FileSystem* fs = NULL;
  Status s = env_->GetFileSystem("/dir/file1", &fs);
  EXPECT_TRUE(s.ok());
  EXPECT_TRUE(fs != NULL);

  s = env_->GetFileSystem("hdfs://dir/file1", &fs);
  EXPECT_TRUE(!s.ok());
}
