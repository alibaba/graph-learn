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
#include <ctime>
#include <iostream>
#include "contrib/knn/index.h"
#include "contrib/knn/index_factory.h"
#include "gtest/gtest.h"

namespace graphlearn {
namespace op {

class IVFPQIndexTest : public ::testing::Test {
protected:
  void SetUp() override {
  }

  void TearDown() override {
  }

  void TestIndex(KnnIndex* index) {
    // Prepare training data
    float* data = new float[1000 * 64];
    int64_t* ids = new int64_t[1000];
    for (int i = 0; i < 1000; ++i) {
      ids[i] = i * 100;
      float value = static_cast<float>(i) / 100.0;
      for (int j = 0; j < 64; ++j) {
        data[i * 64 + j] = value;
      }
    }

    index->Train(1000, data);
    index->Add(1000, data, ids);

    delete [] data;
    delete [] ids;

    // Prepare input data
    int32_t batch_size = 3;
    float* inputs = new float[batch_size * 64];
    for (int i = 0; i < batch_size; ++i) {
      float value = static_cast<float>((i + 1) * 25) / 100.0;
      for (int j = 0; j < 64; ++j) {
        inputs[i * 64 + j] = value;
      }
    }

    int32_t k = 5;
    int64_t* ret_ids = new int64_t[batch_size * k];
    float* ret_distances = new float[batch_size * k];

    index->Search(batch_size, inputs, k, ret_ids, ret_distances);

    // Check results
    for (int i = 0; i < batch_size * k; ++i) {
      int batch_index = i / k;
      int k_index = i % k;
      if (k_index  == 0) {
        std::cout << std::endl;
      }
      std::cout << ret_ids[i] << ' ';
    }
    std::cout << std::endl;

    for (int i = 0; i < batch_size * k; ++i) {
      if (i % k == 0) {
        std::cout << std::endl;
      }
      std::cout << ret_distances[i] << ' ';
    }
    std::cout << std::endl;

    delete [] ret_ids;
    delete [] ret_distances;
  }
};

TEST_F(IVFPQIndexTest, TrainAddSearch) {
  IndexOption option;
  option.index_type = "ivfpq";
  option.dimension = 64;
  // when nlist == nprobe, the result is same with 'flat'
  option.nlist = 2;
  option.nprobe = 2;
  option.m = 8;

  // Get index
  KnnIndex* index = KnnIndexFactory::Create(option);
  ASSERT_TRUE(index != nullptr);

  TestIndex(index);
  delete index;
}

TEST_F(IVFPQIndexTest, GpuTrainAddSearch) {
  IndexOption option;
  option.index_type = "gpu_ivfpq";
  option.dimension = 64;
  // when nlist == nprobe, the result is same with 'flat'
  option.nlist = 2;
  option.nprobe = 2;
  option.m = 8;

  // Get index
  KnnIndex* index = KnnIndexFactory::Create(option);
  ASSERT_TRUE(index != nullptr);

  TestIndex(index);
  delete index;
}

}  // namespace op
}  // namespace graphlearn
