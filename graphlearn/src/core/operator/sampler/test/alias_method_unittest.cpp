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

#include <unordered_set>
#include "core/operator/sampler/alias_method.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::op;  // NOLINT [build/namespaces]

class AliasMethodTest : public ::testing::Test {
protected:
  void SetUp() override {
    dis_table_.push_back(1.2);
    dis_table_.push_back(0.3);
    dis_table_.push_back(0.4);
  }

protected:
  std::vector<double> dis_table_;
};

TEST_F(AliasMethodTest, Sample) {
  AliasMethod alias_method1;
  AliasMethod alias_method2(&dis_table_);
  AliasMethod alias_method3(alias_method2);

  alias_method1 = alias_method2;

  int32_t* res = new int32_t[5];

  std::unordered_set<IdType> neg_set({0, 1, 2});

  alias_method1.Sample(5, res);
  for (int32_t i = 0; i < 5; ++i) {
    EXPECT_TRUE(neg_set.find(res[i]) != neg_set.end());
  }

  alias_method2.Sample(5, res);
  for (int32_t i = 0; i < 5; ++i) {
    EXPECT_TRUE(neg_set.find(res[i]) != neg_set.end());
  }

  alias_method3.Sample(5, res);
  for (int32_t i = 0; i < 5; ++i) {
    EXPECT_TRUE(neg_set.find(res[i]) != neg_set.end());
  }

  delete [] res;
}

