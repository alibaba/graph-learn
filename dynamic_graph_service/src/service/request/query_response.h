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

#ifndef DGS_SERVICE_REQUEST_QUERY_RESPONSE_H_
#define DGS_SERVICE_REQUEST_QUERY_RESPONSE_H_

#include "common/actor_wrapper.h"
#include "common/log.h"
#include "common/typedefs.h"
#include "core/io/record.h"
#include "core/io/record_builder.h"
#include "generated/fbs/query_response_generated.h"
#include "generated/fbs/record_generated.h"

namespace dgs {

namespace execution {
class Dag;
class DagNode;
}

class QueryResponseBuilder : public io::RecordBatchBuilder {
public:
  explicit QueryResponseBuilder(size_t buf_size = 2048)
    : io::RecordBatchBuilder(buf_size) {}

  void Put(OperatorId opid,
           VertexId vid,
           const std::vector<io::Record>& records);
  void Finish() override;

  size_t size() const { return entry_rep_.size(); }
  void Clear() override;

private:
  std::vector<flatbuffers::Offset<EntryRep>> entry_rep_{};
};

class QueryResponse {
public:
  QueryResponse() : rep_(nullptr), buf_() {}
  explicit QueryResponse(act::BytesBuffer&& buf);
  ~QueryResponse() = default;

  QueryResponse(const QueryResponse&) = delete;
  QueryResponse& operator=(const QueryResponse&) = delete;
  QueryResponse(QueryResponse&& other) noexcept;
  QueryResponse& operator=(QueryResponse&& other) noexcept;

  const QueryResponseRep* GetRep() const { return rep_; }

  const char* data() const { return buf_.get(); }
  size_t size() const { return buf_.size(); }

  act::BytesBuffer Release() {
    return std::move(buf_);
  }

  QueryResponse Share() { return QueryResponse{buf_.share()}; }
  QueryResponse Clone() const { return QueryResponse{buf_.clone()}; }

  void dump_to(act::SerializableQueue &qu);  // NOLINT
  static QueryResponse load_from(act::SerializableQueue &qu);  // NOLINT

private:
  const QueryResponseRep* rep_;
  act::BytesBuffer        buf_;
};

}  // namespace dgs

#endif  // DGS_SERVICE_REQUEST_QUERY_RESPONSE_H_
