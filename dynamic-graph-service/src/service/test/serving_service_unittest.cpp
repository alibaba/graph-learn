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

size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
  ((std::string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

class ServingServiceTest : public ::testing::Test {
public:
  ServingServiceTest() = default;
  ~ServingServiceTest() override = default;

protected:
  void SetUp() override {
    // clear all existing storage first.
    ::system("rm -rf estore_* vstore_* subs_table* record_polling_offsets");
  }
  void TearDown() override {}

protected:
  ServiceTestHelper helper_;
};

TEST_F(ServingServiceTest, RunAll) {
  Service service("../../conf/serving_service_ut_options.yml", 0);

  service.Run();
}
