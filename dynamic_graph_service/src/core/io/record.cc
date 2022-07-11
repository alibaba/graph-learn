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

#include "core/io/record.h"

namespace dgs {
namespace io {

Record::Record(actor::BytesBuffer&& buf)
  : buf_(std::move(buf)), rep_(nullptr), view_(rep_) {
  if (buf_.get() != nullptr) {
    rep_ = flatbuffers::GetRoot<RecordRep>(buf_.get());
    view_ = RecordView{rep_};
  }
}

Record::Record(Record&& other) noexcept
  : buf_(std::move(other.buf_)), rep_(other.rep_),
    view_(other.view_) {
  other.rep_ = nullptr;
}

Record& Record::operator=(Record&& other) noexcept {
  if (this != &other) {
    buf_ = std::move(other.buf_);
    rep_ = other.rep_;
    view_ = other.view_;
    other.rep_ = nullptr;
  }
  return *this;
}

RecordBatch::RecordBatch(actor::BytesBuffer&& buf)
  : buf_(std::move(buf)), rep_(nullptr) {
  if (buf_.get() != nullptr) {
    rep_ = flatbuffers::GetRoot<RecordBatchRep>(buf_.get());
  }
}

RecordBatch::RecordBatch(RecordBatch&& other) noexcept
  : buf_(std::move(other.buf_)), rep_(other.rep_) {
  other.rep_ = nullptr;
}

RecordBatch& RecordBatch::operator=(RecordBatch&& other) noexcept {
  if (this != &other) {
    buf_ = std::move(other.buf_);
    rep_ = other.rep_;
    other.rep_ = nullptr;
  }
  return *this;
}

}  // namespace io
}  // namespace dgs
