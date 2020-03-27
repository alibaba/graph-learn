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

#ifndef GRAPHLEARN_CORE_IO_DATA_SLICER_H_
#define GRAPHLEARN_CORE_IO_DATA_SLICER_H_

#include <cstdint>
#include <vector>

namespace graphlearn {
namespace io {

class DataSlicer {
public:
  DataSlicer(int32_t rank, int32_t nrank, int64_t total_size)
      : rank_(rank), nrank_(nrank), total_size_(total_size) {
    Init();
  }

  int64_t Size() const {
    return total_size_;
  }

  int64_t LocalSize() const {
    return all_num_[rank_];
  }

  int64_t LocalStart() const {
    return Begin()[rank_];
  }

  const int64_t* Begin() const {
    return &beg_num_[0];
  }

  int64_t* Begin() {
    return const_cast<int64_t*>(
      static_cast<const DataSlicer&>(*this).Begin());
  }

  const int64_t* All() const {
    return &all_num_[0];
  }

  int64_t* All() {
    return const_cast<int64_t*>(
      static_cast<const DataSlicer&>(*this).All());
  }

private:
  void Init() {
    beg_num_.reserve(nrank_ + 1);
    all_num_.reserve(nrank_);
    beg_num_.resize(nrank_ + 1);
    all_num_.resize(nrank_);

    int64_t avg = total_size_ / nrank_;
    int64_t re = total_size_ % nrank_;

    for (int32_t i = 0; i < nrank_; ++i) {
        if (i < re) {
            all_num_[i] = avg + 1;
        } else {
            all_num_[i] = avg;
        }
    }

    beg_num_[0] = 0;
    for (int32_t i = 1; i <= nrank_; ++i) {
        beg_num_[i] = beg_num_[i-1] + all_num_[i-1];
    }
  }

private:
  int32_t rank_;
  int32_t nrank_;
  int64_t total_size_;
  std::vector<int64_t> beg_num_;
  std::vector<int64_t> all_num_;
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_IO_DATA_SLICER_H_
