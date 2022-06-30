// Copyright (c) 2019, Alibaba Inc.
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

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include "graphlearn/contrib/knn/index.h"
#include "graphlearn/contrib/knn/index_factory.h"
#include "graphlearn/contrib/knn/index_manager.h"
#include "gtest/gtest.h"

namespace graphlearn {
namespace op {

class IndexFactoryManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
  }

  void TearDown() override {
  }
};

TEST_F(IndexFactoryManagerTest, Create) {
  IndexOption option;
  option.index_type = "flat";
  option.dimension = 100;
  option.nlist = 5;
  option.nprobe = 2;
  option.m = 10;

  KnnIndex* index = KnnIndexFactory::Create(option);
  ASSERT_TRUE(index != nullptr);
  delete index;

  option.index_type = "ivfflat";
  index = KnnIndexFactory::Create(option);
  ASSERT_TRUE(index != nullptr);
  delete index;

  option.index_type = "ivfpq";
  index = KnnIndexFactory::Create(option);
  ASSERT_TRUE(index != nullptr);
  delete index;

  option.index_type = "abcd";
  index = KnnIndexFactory::Create(option);
  ASSERT_TRUE(index == nullptr);
}

TEST_F(IndexFactoryManagerTest, AddGetRemove) {
  IndexOption option;
  option.index_type = "flat";
  option.dimension = 100;

  KnnIndex* index = KnnIndexFactory::Create(option);
  ASSERT_TRUE(index != nullptr);

  KnnIndexManager::Instance()->Add("user", index);
  ASSERT_TRUE(KnnIndexManager::Instance()->Get("user") != nullptr);
  ASSERT_TRUE(KnnIndexManager::Instance()->Get("item") == nullptr);

  KnnIndex* index_2 = KnnIndexFactory::Create(option);
  ASSERT_TRUE(index_2 != nullptr);
  KnnIndexManager::Instance()->Add("item", index_2);
  ASSERT_TRUE(KnnIndexManager::Instance()->Get("user") != nullptr);
  ASSERT_TRUE(KnnIndexManager::Instance()->Get("item") != nullptr);

  KnnIndexManager::Instance()->Remove("user");
  ASSERT_TRUE(KnnIndexManager::Instance()->Get("user") == nullptr);

  KnnIndexManager::Instance()->Remove("item");
  ASSERT_TRUE(KnnIndexManager::Instance()->Get("item") == nullptr);
}

}  // namespace op
}  // namespace graphlearn
