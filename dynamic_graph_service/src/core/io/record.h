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

#ifndef DGS_CORE_IO_RECORD_H_
#define DGS_CORE_IO_RECORD_H_

#include "common/actor_wrapper.h"
#include "core/io/record_view.h"
#include "generated/fbs/record_generated.h"

namespace dgs {
namespace io {

/// The structure of io record.
///
/// The underlying buffer is seastar temporary buffer in order to
/// facilitate the construction of a record from different payloads.
class Record {
public:
  Record() : buf_(), rep_(nullptr), view_(rep_) {}
  explicit Record(actor::BytesBuffer&& buf);

  Record(const Record&) = delete;
  Record& operator=(const Record&) = delete;
  Record(Record&& other) noexcept;
  Record& operator=(Record&& other) noexcept;

  /// Get the view of current record.
  const RecordView& GetView() const {
    return view_;
  }

  /// Get the raw data pointer of the underlying buffer.
  const char* Data() const {
    return buf_.get();
  }

  /// Get the raw data size of the underlying buffer.
  size_t Size() const {
    return buf_.size();
  }

  /// Make a new record referring to the same underlying buffer.
  Record Share() {
    return Record{buf_.share()};
  }

  /// Make a new clone of current record.
  Record Clone() const {
    return Record{buf_.clone()};
  }

  /// Get the underlying temporary buffer.
  const actor::BytesBuffer& Buffer() const {
    return buf_;
  }

  /// Release the underlying string buffer of this record.
  ///
  /// \remark After calling this method, the current record
  /// is no longer valid.
  actor::BytesBuffer ReleaseBuffer() {
    return std::move(buf_);
  }

private:
  actor::BytesBuffer buf_;
  const RecordRep*   rep_;
  RecordView         view_;
};

/// The structure of io record batch.
///
/// The record batch is the basic payload of dataloader
/// messages and actor messages.
///
/// The underlying buffer is cppkafka message in order
/// to reduce one copy when constructing a record batch.
class RecordBatch {
public:
  RecordBatch() : buf_(), rep_(nullptr) {}
  explicit RecordBatch(actor::BytesBuffer&& buf);

  RecordBatch(const RecordBatch&) = delete;
  RecordBatch& operator=(const RecordBatch&) = delete;
  RecordBatch(RecordBatch&& other) noexcept;
  RecordBatch& operator=(RecordBatch&& other) noexcept;

  /// Get View of the current record batch.
  RecordBatchView GetView() const {
    return RecordBatchView{rep_};
  }

  /// Get the raw data pointer of the underlying message buffer.
  const char* Data() const {
    return buf_.get();
  }

  /// Get the raw data size of the underlying message buffer.
  size_t Size() const {
    return buf_.size();
  }

  /// Get the underlying buffer.
  const actor::BytesBuffer& Buffer() const {
    return buf_;
  }

  /// Release the underlying buffer.
  ///
  /// \remark After calling this method, the current record
  /// batch is no longer valid
  actor::BytesBuffer ReleaseBuffer() {
    return std::move(buf_);
  }

  /// As each record batch polled from dataloader will be
  /// ingested locally, the \dump_to and \load_from methods
  /// are implemented with null.
  void dump_to(actor::SerializableQueue& qu) {}  // NOLINT
  static RecordBatch load_from(actor::SerializableQueue& qu) {  // NOLINT
    return RecordBatch{};
  }

private:
  actor::BytesBuffer    buf_;
  const RecordBatchRep* rep_;
};

}  // namespace io
}  // namespace dgs

#endif  // DGS_CORE_IO_RECORD_H_
