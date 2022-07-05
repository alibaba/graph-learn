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

#include "common/string/lite_string.h"

#include <algorithm>
#include <iostream>
#include "common/base/hash.h"

namespace graphlearn {

const size_t LiteString::npos = size_t(-1);

std::ostream& operator<<(std::ostream& o, LiteString piece) {
  o.write(piece.data(), piece.size());
  return o;
}

bool LiteString::contains(LiteString s) const {
  return std::search(begin(), end(), s.begin(), s.end()) != end();
}

size_t LiteString::find(char c, size_t pos) const {
  if (pos >= size_) {
    return npos;
  }
  const char* result = reinterpret_cast<const char*>(
    memchr(data_ + pos, c, size_ - pos));
  return result != nullptr ? result - data_ : npos;
}

size_t LiteString::rfind(char c, size_t pos) const {
  if (size_ == 0) {
    return npos;
  }

  const char* p = data_ + std::min(pos, size_ - 1);
  for (; p >= data_; p--) {
    if (*p == c) {
      return p - data_;
    }
  }
  return npos;
}

LiteString LiteString::substr(size_t pos, size_t n) const {
  if (pos > size_) {
    pos = size_;
  }
  if (n > size_ - pos) {
    n = size_ - pos;
  }
  return LiteString(data_ + pos, n);
}

}  // namespace graphlearn
