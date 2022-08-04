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

#ifndef DGS_SERVICE_REQUEST_QUERY_REQUEST_H_
#define DGS_SERVICE_REQUEST_QUERY_REQUEST_H_

#include "common/actor_wrapper.h"
#include "common/typedefs.h"
#include "generated/fbs/install_query_req_generated.h"
#include "generated/fbs/run_query_req_generated.h"
#include "generated/fbs/uninstall_query_req_generated.h"

namespace dgs {

class InstallQueryRequest {
public:
  explicit InstallQueryRequest(act::BytesBuffer&& buf,
                               bool is_chief = false);
  InstallQueryRequest(const InstallQueryRequest&) = delete;
  InstallQueryRequest(InstallQueryRequest&& other);

  ~InstallQueryRequest() = default;

  QueryPriority priority() const {
    return rep_->priority();
  }

  const QueryPlanRep* query_plan() const {
    return rep_->query_plan();
  }

  QueryId query_id() const {
    return rep_->query_id();
  }

  bool is_chief() const {
    return static_cast<bool>(is_chief_);
  }

  act::BytesBuffer CloneBuffer() const {
    return buf_.clone();
  }

  void dump_to(act::SerializableQueue& qu);  // NOLINT
  static InstallQueryRequest load_from(act::SerializableQueue& qu);  // NOLINT

private:
  // flag `is_chief_` type is uint8_t instead of bool due to
  // sizeof(bool) is implementation defined, which may cause
  // platform incompatibility issues.
  const uint8_t          is_chief_;
  act::BytesBuffer       buf_;
  InstallQueryRequestRep *rep_;
};

class UnInstallQueryRequest {
public:
  explicit UnInstallQueryRequest(act::BytesBuffer &&buf,
                                 bool is_chief = false);
  UnInstallQueryRequest(const UnInstallQueryRequest&) = delete;
  UnInstallQueryRequest(UnInstallQueryRequest&& other);

  ~UnInstallQueryRequest() = default;

  QueryId query_id() const {
    return rep_->query_id();
  }

  bool is_chief() const {
    return static_cast<bool>(is_chief_);
  }

  act::BytesBuffer CloneBuffer() const {
    return buf_.clone();
  }

  void dump_to(act::SerializableQueue& qu);  // NOLINT
  static UnInstallQueryRequest load_from(act::SerializableQueue& qu);  // NOLINT

private:
  // flag `is_chief_` type is uint8_t instead of bool due to
  // sizeof(bool) is implementation defined, which may cause
  // platform incompatibility issues.
  const uint8_t            is_chief_;
  act::BytesBuffer         buf_;
  UnInstallQueryRequestRep *rep_;
};

class RunQueryRequest {
public:
  RunQueryRequest() {}
  RunQueryRequest(QueryId qid, VertexId vid);
  RunQueryRequest(const RunQueryRequest&) = delete;
  RunQueryRequest(RunQueryRequest&& other);

  ~RunQueryRequest() = default;

  QueryId query_id() const {
    return qid_;
  }

  VertexId vid() const {
    return vid_;
  }

  void dump_to(act::SerializableQueue& qu);  // NOLINT
  static RunQueryRequest load_from(act::SerializableQueue& qu);  // NOLINT

private:
  QueryId  qid_;
  VertexId vid_;
};

}  // namespace dgs

#endif  // DGS_SERVICE_REQUEST_QUERY_REQUEST_H_
