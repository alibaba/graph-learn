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

#include "graphlearn/common/base/hash.h"

#include <cstring>

namespace graphlearn {

namespace {

const bool kLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;

inline uint32_t ByteAs32(char c) {
  return static_cast<uint32_t>(c) & 0xff;
}

inline uint64_t ByteAs64(char c) {
  return static_cast<uint64_t>(c) & 0xff;
}

uint32_t DecodeAs32(const char* ptr) {
  if (kLittleEndian) {
    uint32_t result = 0;
    ::memcpy(&result, ptr, sizeof(result));
    return result;
  } else {
    return ((static_cast<uint32_t>(static_cast<unsigned char>(ptr[0]))) |
            (static_cast<uint32_t>(static_cast<unsigned char>(ptr[1])) << 8) |
            (static_cast<uint32_t>(static_cast<unsigned char>(ptr[2])) << 16) |
            (static_cast<uint32_t>(static_cast<unsigned char>(ptr[3])) << 24));
  }
}

uint64_t DecodeAs64(const char* ptr) {
  if (kLittleEndian) {
    uint64_t result = 0;
    ::memcpy(&result, ptr, sizeof(result));
    return result;
  } else {
    uint64_t l = DecodeAs32(ptr);
    uint64_t h = DecodeAs32(ptr + 4);
    return (h << 32) | l;
  }
}

}  // anonymous namespace

uint32_t Hash32(const char* data, size_t n, uint32_t seed) {
  // 'm' and 'r' are what we refer to the ones in TensorFlow.
  const uint32_t m = 0x5bd1e995;
  const uint32_t r = 24;

  uint32_t h = seed ^ n;

  while (n >= 4) {
    uint32_t k = DecodeAs32(data);

    k *= m;
    k ^= k >> r;
    k *= m;

    h *= m;
    h ^= k;

    data += 4;
    n -= 4;
  }

  switch (n) {
    case 3:
      h ^= ByteAs32(data[2]) << 16;
    case 2:
      h ^= ByteAs32(data[1]) << 8;
    case 1:
      h ^= ByteAs32(data[0]);
      h *= m;
  }

  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;

  return h;
}

uint64_t Hash64(const char* data, size_t n, uint64_t seed) {
  // 'm' and 'r' are what we refer to the ones in TensorFlow.
  const uint64_t m = 0xc6a4a7935bd1e995;
  const uint32_t r = 47;

  uint64_t h = seed ^ (n * m);

  while (n >= 8) {
    uint64_t k = DecodeAs64(data);
    data += 8;
    n -= 8;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  switch (n) {
    case 7:
      h ^= ByteAs64(data[6]) << 48;
    case 6:
      h ^= ByteAs64(data[5]) << 40;
    case 5:
      h ^= ByteAs64(data[4]) << 32;
    case 4:
      h ^= ByteAs64(data[3]) << 24;
    case 3:
      h ^= ByteAs64(data[2]) << 16;
    case 2:
      h ^= ByteAs64(data[1]) << 8;
    case 1:
      h ^= ByteAs64(data[0]);
      h *= m;
  }

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

uint64_t Hash64(const char* data, size_t n) {
  return Hash64(data, n, 0xDECAFCAFFE);
}

uint64_t Hash64(const std::string& str) {
  return Hash64(str.data(), str.size());
}

uint32_t Hash32(const char* data, size_t n) {
  return Hash32(data, n, 0xCAFFE);
}

uint32_t Hash32(const std::string& str) {
  return Hash32(str.data(), str.size());
}

}  // namespace graphlearn
