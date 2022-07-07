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

#include "common/base/time_stamp.h"

#include <sys/time.h>

namespace graphlearn {

int64_t GetTimeStampInMs() {
  int64_t time = static_cast<int64_t>(GetTimeStampInUs() * 0.001 + 0.01);
  return time;
}

int64_t GetTimeStampInUs() {
  struct timeval tv;
  ::gettimeofday(&tv, 0);
  int64_t time = tv.tv_sec * 1000000 + tv.tv_usec;
  return time;
}

}  // namespace graphlearn
