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

#ifndef GRAPHLEARN_COMMON_BASE_PROGRESS_H_
#define GRAPHLEARN_COMMON_BASE_PROGRESS_H_

#include <cstdio>
#include <iostream>
#include <mutex>  //NOLINT [build/c++11]
#include <string>
#include "common/base/log.h"
#include "common/threading/sync/lock.h"

#define PROGRESSING(key)                   \
  static ::graphlearn::profiling::Progress key##_progress(#key)

#define PROGRESSINGG(key, g)               \
  static ::graphlearn::profiling::Progress key##_progress(#key, g)

#define UPDATE_PROGRESSING(key, counter)   \
  key##_progress.Add(counter)

namespace graphlearn {
namespace profiling {

struct Progress {
  std::string      key;
  std::mutex       mtx_;
  volatile int64_t grading;
  volatile int64_t count;
  volatile int64_t stage;

  explicit Progress(const std::string& k, int64_t g = 1000000)
    : key(k), grading(g), count(0), stage(1) {
  }

  ~Progress() {
  }

  void Add(int64_t counter) {
    ScopedLocker<std::mutex> _(&mtx_);
    count += counter;
    if (count > stage * grading) {
      char buffer[100];
      snprintf(buffer, sizeof(buffer),
               "Progress of %s: %ld", key.c_str(), stage * grading);
      USER_LOG(buffer);
      ++stage;
    }
  }
};

}  // namespace profiling
}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_BASE_PROGRESS_H_
