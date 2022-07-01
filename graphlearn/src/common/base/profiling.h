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

#ifndef GRAPHLEARN_COMMON_BASE_PROFILING_H_
#define GRAPHLEARN_COMMON_BASE_PROFILING_H_

#include <iostream>
#include <string>
#include <unordered_map>
#include "graphlearn/common/base/time_stamp.h"

#if defined(OPEN_PROFILING)
#define PROFILING(key)                                      \
  static ::graphlearn::profiling::Costage key##_cost(#key); \
  ::graphlearn::profiling::Timer key##_timer(&key##_cost);
#else
#define PROFILING(key)
#endif

namespace graphlearn {
namespace profiling {

struct Costage {
  std::string key;
  int32_t     latency;
  int32_t     count;

  explicit Costage(const std::string& k)
    : key(k), latency(0), count(0) {
  }

  ~Costage() {
    std::cout << "profiling key: " << key
              << ", latency: " << static_cast<double>(latency) / 1000.0
              << " ms, count: " << count << std::endl;
  }

  void Add(int32_t t) {
    latency += t;
    ++count;
  }
};

class Timer {
private:
  Costage* cost_;
  int64_t  begin_;

public:
  explicit Timer(Costage* cost)
    : cost_(cost) {
    begin_ = GetTimeStampInUs();
  }

  ~Timer() {
    cost_->Add(GetTimeStampInUs() - begin_);
  }
};

}  // namespace profiling
}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_BASE_PROFILING_H_
