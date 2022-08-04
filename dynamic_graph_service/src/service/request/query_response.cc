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

#include "service/request/query_response.h"

#include "core/io/record_view.h"

namespace dgs {

void QueryResponseBuilder::Put(OperatorId opid,
                               VertexId vid,
                               const std::vector<io::Record>& records) {
  for (auto& record : records) {
    AddRecord(record.GetView());
  }
  auto flat_records = builder_.CreateVector(record_vec_);
  auto batch = CreateRecordBatchRep(builder_, flat_records);
  entry_rep_.emplace_back(CreateEntryRep(builder_, opid, vid, batch));
  record_vec_.clear();
}

void QueryResponseBuilder::Finish() {
  auto flat_entries = builder_.CreateVector(entry_rep_);
  auto entries = CreateQueryResponseRep(builder_, flat_entries);
  builder_.Finish(entries);
}

void QueryResponseBuilder::Clear() {
  builder_.Clear();
  record_vec_.clear();
  entry_rep_.clear();
}

QueryResponse::QueryResponse(act::BytesBuffer&& buf)
  : buf_(std::move(buf)), rep_(nullptr) {
  if (buf_.get() != nullptr) {
    rep_ = flatbuffers::GetRoot<QueryResponseRep>(buf_.get());
  }
}

QueryResponse::QueryResponse(QueryResponse&& other) noexcept
  : buf_(std::move(other.buf_)), rep_(other.rep_) {
  other.rep_ = nullptr;
}

QueryResponse& QueryResponse::operator=(QueryResponse&& other) noexcept {
  if (this != &other) {
    buf_ = std::move(other.buf_);
    rep_ = other.rep_;
    other.rep_ = nullptr;
  }
  return *this;
}

void QueryResponse::dump_to(act::SerializableQueue& qu) {
  qu.push(std::move(buf_));
}

QueryResponse QueryResponse::load_from(act::SerializableQueue& qu) {
  return QueryResponse(qu.pop());
}

}  // namespace dgs
