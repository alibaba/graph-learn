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

#ifndef GRAPHLEARN_COMMON_STRING_LITE_STRING_H_
#define GRAPHLEARN_COMMON_STRING_LITE_STRING_H_

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iosfwd>
#include <string>

namespace graphlearn {

class LiteString {
public:
  typedef const char* const_iterator;
  typedef const char* iterator;

  static const size_t npos;

  LiteString() : data_(""), size_(0) {}

  LiteString(const char* s, size_t n) : data_(s), size_(n) {}

  LiteString(const char* s)  //NOLINT
      : data_(s), size_(strlen(s)) {
  }

  LiteString(const std::string& s)  //NOLINT
      : data_(s.data()), size_(s.size()) {
  }

  void set(const void* data, size_t len) {
    data_ = reinterpret_cast<const char*>(data);
    size_ = len;
  }

  const char* data() const {
    return data_;
  }

  size_t size() const {
    return size_;
  }

  bool empty() const {
    return size_ == 0;
  }

  iterator begin() const {
    return data_;
  }

  iterator end() const {
    return data_ + size_;
  }

  char operator[](size_t n) const {
    assert(n < size());
    return data_[n];
  }

  void clear() {
    data_ = "";
    size_ = 0;
  }

  void remove_prefix(size_t n) {
    assert(n <= size());
    data_ += n;
    size_ -= n;
  }

  void remove_suffix(size_t n) {
    assert(size_ >= n);
    size_ -= n;
  }

  bool Consume(LiteString x) {
    if (starts_with(x)) {
      remove_prefix(x.size_);
      return true;
    }
    return false;
  }

  bool Consume(const char* x) {
    return Consume(LiteString(x));
  }

  bool starts_with(LiteString x) const {
    return ((size_ >= x.size_) &&
            (memcmp(data_, x.data_, x.size_) == 0));
  }

  bool ends_with(LiteString x) const {
    return ((size_ >= x.size_) &&
            (memcmp(data_ + (size_ - x.size_), x.data_, x.size_) == 0));
  }

  std::string ToString() const {
    return std::string(data_, size_);
  }

  //   <  0 if "*this" <  "b",
  //   == 0 if "*this" == "b",
  //   >  0 if "*this" >  "b"
  int32_t compare(LiteString b) const {
    const size_t min_len = (size_ < b.size_) ? size_ : b.size_;
    int r = memcmp(data_, b.data_, min_len);
    if (r == 0) {
      if (size_ < b.size_)
        r = -1;
      else if (size_ > b.size_)
        r = +1;
    }
    return r;
  }

  bool contains(LiteString s) const;
  LiteString substr(size_t pos, size_t n = npos) const;
  size_t find(char c, size_t pos = 0) const;
  size_t rfind(char c, size_t pos = npos) const;

private:
  const char* data_;
  size_t      size_;
};

inline bool operator==(LiteString x, LiteString y) {
  return ((x.size() == y.size()) &&
          (memcmp(x.data(), y.data(), x.size()) == 0));
}

inline bool operator!=(LiteString x, LiteString y) {
  return !(x == y);
}

inline bool operator<(LiteString x, LiteString y) {
  return x.compare(y) < 0;
}

inline bool operator>(LiteString x, LiteString y) {
  return x.compare(y) > 0;
}

inline bool operator<=(LiteString x, LiteString y) {
  return x.compare(y) <= 0;
}

inline bool operator>=(LiteString x, LiteString y) {
  return x.compare(y) >= 0;
}

extern std::ostream& operator<<(std::ostream& o, LiteString piece);

}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_STRING_LITE_STRING_H_
