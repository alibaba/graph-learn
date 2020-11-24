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

#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_TYPES_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_TYPES_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#if __cplusplus >= 201103L
#include <unordered_map>
#else
#include <tr1/unordered_map>  // NOLINT [build/c++tr1]
#endif

#include "graphlearn/core/io/element_value.h"

namespace graphlearn {
namespace io {

typedef int64_t IdType;
typedef int32_t IndexType;

typedef std::vector<IdType> IdList;
typedef std::vector<IndexType> IndexList;
typedef std::vector<std::vector<IdType>> IdMatrix;
#if __cplusplus >= 201103L
typedef std::unordered_map<IdType, IndexType> MAP;
#else
typedef std::tr1::unordered_map<IdType, IndexType> MAP;
#endif

template <class T>
class MultiArray {
public:
  MultiArray(const std::vector<const T*>& values, const std::vector<int32_t> sizes,
             int32_t element_size, int32_t element_offset=0)
      : values_(values),
        sizes_(sizes),
        element_size_(element_size),
        element_offset_(element_offset) {
    offsets_.emplace_back(0);
    for (size_t i = 1; i <= sizes.size(); ++i) {
      offsets_.emplace_back(offsets_[i-1] + sizes[i-1]);
    }
  }
  MultiArray(MultiArray&& rhs) {
    values_ = rhs.values_;
    sizes_ = rhs.sizes_;
    offsets_ = rhs.offsets_;
    element_size_ = rhs.element_size_;
    element_offset_ = rhs.element_offset_;
  }
  operator bool () const {
    return *offsets_.rbegin() > 0;
  }
  T operator[] (int32_t i) const {
    auto p = std::upper_bound(offsets_.begin(), offsets_.end(), i);
    if (p == offsets_.end()) {
      throw std::out_of_range("Index out of range: " + std::to_string(i));
    }
    int idx = p - offsets_.begin();
    auto target = reinterpret_cast<const uint8_t *>(values_[idx-1])
                + (element_size_ * (i-offsets_[idx-1]))
                + element_offset_;
    return *reinterpret_cast<const T *>(target);
  }
  int32_t Size() const {
    return *offsets_.rbegin();
  }
private:
  std::vector<const T *> values_;
  std::vector<int32_t> sizes_;
  std::vector<int32_t> offsets_;
  int32_t element_size_;
  int32_t element_offset_;
};

template <typename T>
class RangeArray {
public:
  RangeArray(T const &begin, T const &end): begin_(begin), end_(end) {}

  virtual operator bool () const {
    return begin_ == end_;
  }

  virtual T operator[] (int32_t i) const {
    return begin_ + i;
  }

  virtual int32_t Size() const {
    return end_ - begin_;
  }

private:
  T const begin_;
  T const end_;
};

template <class T>
class Array {
public:
  Array() : value_(nullptr), mvalue_(nullptr), size_(0) {
  }

  Array(std::shared_ptr<MultiArray<T>> const &mvalue)
    : value_(nullptr), mvalue_(mvalue), size_(mvalue->Size()) {
  }

  Array(const T* value, int32_t size)
    : value_(value), size_(size) {
  }

  Array(T const &begin, T const &end) :
      value_(nullptr),
      mvalue_(nullptr),
      rangevalue_(std::make_shared<RangeArray<T>>(begin, end)),
      size_(end - begin) {
  }

  explicit Array(const std::vector<T>& values)
    : value_(values.data()), size_(values.size()) {
  }

  Array(const Array& rhs) {
    value_ = rhs.value_;
    mvalue_ = rhs.mvalue_;
    size_ = rhs.size_;
  }

  Array(Array&& rhs) {
    value_ = rhs.value_;
    mvalue_ = rhs.mvalue_;
    size_ = rhs.size_;
  }

  virtual ~Array() {
  }

  virtual operator bool () const {
    return value_ != nullptr && size_ != 0;
  }

  virtual T operator[] (int32_t i) const {
    if (mvalue_) {
      return mvalue_->operator[](i);
    }
    if (rangevalue_) {
      return rangevalue_->operator[](i);
    }
    return value_[i];
  }

  T at(int32_t i) const {
    return this->operator[](i);
  }

  virtual int32_t Size() const {
    return size_;
  }

private:
  const T* value_;
  std::shared_ptr<MultiArray<T>> mvalue_;
  std::shared_ptr<RangeArray<T>> rangevalue_;
  int32_t size_;
};

typedef Array<IdType> IdArray;

class Attribute {
public:
  Attribute() : value_(nullptr), own_(false) {
  }

  Attribute(AttributeValue* value, bool own)
    : value_(value), own_(own) {
  }

  Attribute(Attribute&& rhs) {
    value_ = rhs.value_;
    own_ = rhs.own_;
    rhs.value_ = nullptr;
    rhs.own_ = false;
  }

  ~Attribute() {
    if (own_) {
      delete value_;
    }
  }

  AttributeValue* operator -> () const {
    return value_;
  }

  AttributeValue* get() const {
    return value_;
  }

private:
  AttributeValue* value_;
  bool own_;
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_TYPES_H_

