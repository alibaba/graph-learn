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

#ifndef GRAPHLEARN_CONTRIB_KNN_HEAP_H_
#define GRAPHLEARN_CONTRIB_KNN_HEAP_H_

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace graphlearn {
namespace op {

struct MinCompare {
  virtual bool operator() (float left, float right) {
    return left < right;
  }
};

struct MaxCompare {
  virtual bool operator() (float left, float right) {
    return left > right;
  }
};

template<class T, class Compare = MinCompare>
class Heap {
public:
  explicit Heap(int32_t capacity) : capacity_(capacity), size_(0) {
    value_.resize(capacity);
    attachment_.resize(capacity);
  }

  void Push(float value, const T& attach) {
    if (size_ < capacity_) {
      value_[size_] = value;
      attachment_[size_] = attach;
      ++size_;
      UpwardAdjust();
    } else if (comp_(value, value_[0])) {
      value_[0] = value;
      attachment_[0] = attach;
      DownwardAdjust();
    } else {
      // do nothing
    }
  }

  bool Pop(float* value, T* attach) {
    if (Empty()) {
      return false;
    }

    *value = value_[0];
    *attach = attachment_[0];
    std::swap(value_[0], value_[size_ - 1]);
    std::swap(attachment_[0], attachment_[size_ - 1]);

    --size_;
    DownwardAdjust();
    return true;
  }

  int32_t Size() {
    return size_;
  }

  bool Empty() {
    return size_ == 0;
  }

  void Clear() {
    size_ = 0;
  }

private:
  void UpwardAdjust() {
    int32_t current = size_ - 1;
    while (current > 0) {
      int32_t parent = (current - 1) / 2;
      if (comp_(value_[parent], value_[current])) {
        std::swap(value_[parent], value_[current]);
        std::swap(attachment_[parent], attachment_[current]);
      }
      current = parent;
    }
  }

  void DownwardAdjust() {
    int32_t current = 0;
    while (current < size_) {
      int32_t left = current * 2 + 1;
      int32_t right = current * 2 + 2;
      if (left >= size_) {
        break;
      }

      int32_t target = current;
      if (comp_(value_[target], value_[left])) {
        target = left;
      }
      if (right < size_) {
        if (comp_(value_[target], value_[right])) {
          target = right;
        }
      }

      if (target == current) {
        break;
      } else {
        std::swap(value_[target], value_[current]);
        std::swap(attachment_[target], attachment_[current]);
        current = target;
      }
    }
  }

private:
  int32_t            capacity_;
  int32_t            size_;
  std::vector<float> value_;
  std::vector<T>     attachment_;
  Compare            comp_;
};

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CONTRIB_KNN_HEAP_H_
