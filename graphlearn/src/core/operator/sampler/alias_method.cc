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

#include "core/operator/sampler/alias_method.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <random>
#include <unordered_set>

namespace graphlearn {
namespace op {

AliasMethod::AliasMethod() : range_(0) {
}

AliasMethod::AliasMethod(const std::vector<float>* dist)
    : range_(dist->size()) {
  Build(dist);
}

AliasMethod::AliasMethod(int32_t uniform_max) : range_(uniform_max) {
  std::vector<float> dist(uniform_max, 1.0);
  Build(&dist);
}

AliasMethod::AliasMethod(const AliasMethod& rhs) {
  range_ = rhs.range_;
  alias_ = rhs.alias_;
  probs_ = rhs.probs_;
}

AliasMethod& AliasMethod::operator=(const AliasMethod& rhs) {
  if (&rhs == this) {
    return *this;
  }

  range_ = rhs.range_;
  alias_ = rhs.alias_;
  probs_ = rhs.probs_;
  return *this;
}

void AliasMethod::Build(const std::vector<float>* dist) {
  int32_t count = dist->size();
  if (count == 0) {
    return;
  }

  alias_.resize(count);
  probs_.resize(count);

  std::vector<int32_t> high_set;
  std::vector<int32_t> low_set;
  high_set.reserve(count / 2 + 1);
  low_set.reserve(count / 2 + 1);

  // initialize.
  float avg_prob = 1.0 / count;
  float sum = std::accumulate(dist->begin(), dist->end(), 0.0);
  for (int32_t i = 0; i < count; i++) {
    alias_[i] = i;
    float prob = (*dist)[i] / sum;
    probs_[i] = prob * count;
    if (prob < avg_prob) {
      low_set.push_back(i);
    } else if (prob > avg_prob) {
      high_set.push_back(i);
    }
  }

  // update.
  int32_t low_num = low_set.size();
  int32_t high_num = high_set.size();
  while (low_num > 0 && high_num > 0) {
    int32_t low_idx = low_set[--low_num];
    int32_t high_idx = high_set[--high_num];
    probs_[high_idx] = probs_[high_idx] - 1 + probs_[low_idx];
    alias_[low_idx] = high_idx;
    if (probs_[high_idx] < 1.0) {
      low_set[low_num++] = high_idx;
    } else if (probs_[high_idx] > 1.0) {
      high_set[high_num++] = high_idx;
    }
  }

  while (low_num > 0) {
    probs_[low_set[--low_num]] = 1.0;
  }

  while (high_num > 0) {
    probs_[high_set[--high_num]] = 1.0;
  }
}

bool AliasMethod::Sample(int32_t num, int32_t* ret) {
  if (range_ == 0) {
    return false;
  }

  thread_local static std::random_device rd;
  thread_local static std::mt19937 engine(rd());

  std::uniform_real_distribution<> generator(0, range_ - 1);
  for (int32_t i = 0; i < num; i++) {
    float rand = generator(engine);
    int32_t idx = static_cast<int32_t>(rand);
    ret[i] = (probs_[idx] <= (rand - idx)) ? alias_[idx] : idx;
  }
  return true;
}

}  // namespace op
}  // namespace graphlearn
