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

#include "service/service.h"
#include "gtest/gtest.h"

using namespace dgs;

class SamplingServiceTest : public ::testing::Test {
public:
  SamplingServiceTest() = default;
  ~SamplingServiceTest() override = default;

protected:
  void SetUp() override {
    ::system("rm -rf ./tmp_store && mkdir -p ./tmp_store");
    ::system("rm -rf ./tmp_subs_table && mkdir -p ./tmp_subs_table");
  }
  void TearDown() override {}
};

TEST_F(SamplingServiceTest, RunAll) {
  Service service("../../conf/ut/sampling.ut.yml", 0);
  FLAGS_alsologtostderr = true;
  service.Run();
}
