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

#include "common/io/line_iterator.h"

#include <algorithm>
#include "common/base/errors.h"

namespace graphlearn {
namespace io {

LineIterator::LineIterator(ByteStreamAccessFile* file,
                           size_t buffer_bytes)
    : file_(file),
      size_(buffer_bytes),
      buf_(new char[size_]),
      pos_(buf_),
      limit_(buf_) {
}

LineIterator::~LineIterator() {
  delete[] buf_;
}

Status LineIterator::Next(std::string* result) {
  result->clear();
  Status s;
  do {
    size_t buf_remain = limit_ - pos_;
    char* newline = static_cast<char*>(memchr(pos_, '\n', buf_remain));
    if (newline != nullptr) {
      size_t result_len = newline - pos_;
      result->append(pos_, result_len);
      pos_ = newline + 1;
      if (!result->empty() && result->back() == '\r') {
        result->resize(result->size() - 1);
      }
      return Status::OK();
    }
    if (buf_remain > 0) {
      result->append(pos_, buf_remain);
    }
    s = FillBuffer();
  } while (limit_ != buf_);

  if (!result->empty() && result->back() == '\r') {
    result->resize(result->size() - 1);
  }
  if (error::IsOutOfRange(s) && !result->empty()) {
    return Status::OK();
  }
  return s;
}

Status LineIterator::FillBuffer() {
  LiteString data;
  Status s = file_->Read(size_, &data, buf_);

  if (data.data() != buf_) {
    memmove(buf_, data.data(), data.size());
  }
  pos_ = buf_;
  limit_ = pos_ + data.size();
  return s;
}

}  // namespace io
}  // namespace graphlearn
