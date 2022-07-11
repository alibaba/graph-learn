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

using namespace dgs;

class ServingActorModuleTest : public ::testing::Test {
public:
  ServingActorModuleTest() : helper_(4, 4, 2) {}
  ~ServingActorModuleTest() override = default;

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

TEST_F(ServingActorModuleTest, RunQuery) {
  helper_.InstallQuery();
  helper_.MakeSampleStore();

  auto fut = seastar::alien::submit_to(
      *seastar::alien::internal::default_instance, 0, [this] {
    return helper_.GetServingActorRef(0).RunQuery(
        ServingTestHelper::MakeRunQueryRequest(2));
  });
  fut.wait();

  ServingTestHelper::PrintQueryResponse(fut.get());
}
