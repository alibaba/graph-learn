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

#ifndef DGS_COMMON_UTILS_H_
#define DGS_COMMON_UTILS_H_

#include <chrono>
#include <cstdint>

namespace dgs {

inline uint64_t CurrentTimeInMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline uint64_t CurrentTimeInUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

template<typename T, typename U>
constexpr size_t OffsetOf(U T::*member) {
    T* obj_ptr = nullptr;
    return reinterpret_cast<char*>(&(obj_ptr->*member)) - \
           reinterpret_cast<char*>(obj_ptr);
}

}  // namespace dgs

#endif  // DGS_COMMON_UTILS_H_
