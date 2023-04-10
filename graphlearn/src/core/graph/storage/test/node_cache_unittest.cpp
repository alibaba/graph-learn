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

#include <iostream>
#include <string>
#include <type_traits>
#include "common/base/log.h"
#include "core/graph/storage/node_cache.h"
#include "core/io/element_value.h"
#include "include/config.h"
#include "gtest/gtest.h"


using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::io;  // NOLINT [build/namespaces]

class NodeCacheTest : public ::testing::Test {

protected:
};

TEST_F(NodeCacheTest, InitTest) {
  LFUCachePolicy<std::string> policy;
  NodeCache<std::string, std::string, LFUCachePolicy> cache(2, std::move(policy));

  cache.Insert("a", "value_a");
  cache.Insert("b", "value_b");

  std::string value;
  bool ret = cache.TryGet("a", value);
  ASSERT_TRUE(ret) <<"cache failed get a value";
  ASSERT_EQ("value_a", value);

  ret = cache.TryGet("b", value);
  ASSERT_TRUE(ret) <<"cache failed get a value";
  ASSERT_EQ("value_b", value);

  ret = cache.TryGet("c", value);
  ASSERT_FALSE(ret) <<"cache failed get a value";
}

TEST_F(NodeCacheTest, EliminateLastInsertElement) {
  LFUCachePolicy<std::string> policy;
  NodeCache<std::string, std::string, LFUCachePolicy> cache(2, std::move(policy));

  cache.Insert("a", "value_a");
  cache.Insert("b", "value_b");
  cache.Insert("c", "value_c");

  std::string value;
  bool ret = cache.TryGet("a", value);
  ASSERT_FALSE(ret) <<"cache failed get a value";
}

TEST_F(NodeCacheTest, EliminateLastAccessElement) {
  LFUCachePolicy<std::string> policy;
  NodeCache<std::string, std::string, LFUCachePolicy> cache(3, std::move(policy));

  cache.Insert("a", "value_a");
  cache.Insert("b", "value_b");
  cache.Insert("c", "value_c");

  std::string value;
  bool ret = cache.TryGet("a", value);
  ASSERT_TRUE(ret) <<"cache failed get a value";

  cache.Insert("d", "value_d");

  ret = cache.TryGet("b", value);
  ASSERT_FALSE(ret) <<"cache failed get a value";
}

TEST_F(NodeCacheTest, InsertElementTwiceIntoCacheTest) {
  LFUCachePolicy<std::string> policy;
  NodeCache<std::string, std::string, LFUCachePolicy> cache(3, std::move(policy));

  cache.Insert("a", "value_a");
  cache.Insert("b", "value_b");
  cache.Insert("c", "value_c");
  cache.Insert("a", "new_value_a");

  std::string value;
  bool ret = cache.TryGet("a", value);
  ASSERT_TRUE(ret) <<"cache failed get a value";
  EXPECT_EQ("new_value_a", value);

  cache.Insert("d", "value_d");
  ret = cache.TryGet("b", value);
  ASSERT_FALSE(ret) <<"cache failed get a value";
}

TEST_F(NodeCacheTest, InsertNodeValueTest) {
  LFUCachePolicy<int> policy;
  NodeCache<int, NodeValue, LFUCachePolicy> cache(3, std::move(policy));

  NodeValue a;
  a.weight = 1.0;
  a.label  = 1;
  int64_t i_attr = 1;
  a.attrs->Add(i_attr);
  a.attrs->Add(i_attr);
  a.attrs->Add(std::string("abccc"));
  a.attrs->Add(std::string("abccc"));

  cache.Insert(1, a);
  cache.Insert(2, a);
  cache.Insert(3, a);
  cache.Insert(4, a);
  cache.Insert(5, a);

  NodeValue value;
  bool ret = cache.TryGet(3, value);
  ASSERT_TRUE(ret) <<"cache failed get a value";
}