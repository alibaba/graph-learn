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

#ifndef GRAPHLEARN_COMMON_BASE_HASH_H_
#define GRAPHLEARN_COMMON_BASE_HASH_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace graphlearn {

uint32_t Hash32(const char* data, size_t n, uint32_t seed);
uint64_t Hash64(const char* data, size_t n, uint64_t seed);
uint32_t Hash32(const char* data, size_t n);
uint64_t Hash64(const char* data, size_t n);
uint32_t Hash32(const std::string& str);
uint64_t Hash64(const std::string& str);

template <typename T>
struct Hash {
  size_t operator()(const T& t) const { return std::hash<T>()(t); }
};

template <>
struct Hash<std::string> {
  size_t operator()(const std::string& s) const {
    return static_cast<size_t>(Hash64(s));
  }
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_BASE_HASH_H_
