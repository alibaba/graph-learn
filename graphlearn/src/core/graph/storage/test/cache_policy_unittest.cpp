/* Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.

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

#include <string>
#include "common/base/log.h"
#include "core/graph/storage/cache_policy.h"
#include "include/config.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::io;  // NOLINT [build/namespaces]

class LFUCachePolicyTest : public ::testing::Test {

protected:
  LFUCachePolicy<std::string> policy;
};


TEST_F(LFUCachePolicyTest, OneElementTest) {
  policy.Insert("a");

  EXPECT_EQ("a", policy.Eliminate());
}

TEST_F(LFUCachePolicyTest, VisitTwiceTest) {
  policy.Insert("a");
  policy.Insert("b");
  policy.Visit("a");
  policy.Visit("a");

  EXPECT_EQ("b", policy.Eliminate());
}

TEST_F(LFUCachePolicyTest, VisitSameTwiceTest) {
  policy.Insert("a");
  policy.Insert("b");
  policy.Visit("a");
  policy.Visit("b");

  EXPECT_EQ("a", policy.Eliminate());
}


TEST_F(LFUCachePolicyTest, ElementDeletedTest) {
  policy.Insert("a");
  policy.Insert("b");
  policy.Visit("a");
  policy.Visit("b");
  policy.Erase("b");

  EXPECT_EQ("a", policy.Eliminate());
}