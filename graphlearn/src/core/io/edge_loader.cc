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

#include "core/io/edge_loader.h"

#include <string>
#include <utility>
#include "common/base/errors.h"
#include "common/base/log.h"
#include "core/io/parser.h"
#include "core/io/slice_reader.h"
#include "platform/env.h"

namespace graphlearn {
namespace io {

namespace {

const char* kPattern = "Invalid edge table schema, expected: %s, but got: %s";

}  // anonymous namespace

EdgeLoader::EdgeLoader(const std::vector<EdgeSource>& source,
                       Env* env,
                       int32_t thread_id,
                       int32_t thread_num)
    : source_(nullptr),
      need_resize_(false) {
  reader_ = new SliceReader<EdgeSource>(
    source, env, thread_id, thread_num);
}

EdgeLoader::~EdgeLoader() {
  delete reader_;
}

Status EdgeLoader::Read(EdgeValue* value) {
  Status s = reader_->Read(&record_);
  if (error::IsOutOfRange(s)) {
    LOG(INFO) << "Current edge file completed, " << source_->path;
    return s;
  } else if (!s.ok()) {
    LOG(ERROR) << "Read edge failed, " << s.ToString();
    return s;
  }

  if (need_resize_) {
    value->attrs->Reserve(side_info_.i_num, side_info_.f_num, side_info_.s_num);
    need_resize_ = false;
  }

  s = ParseValue(value);
  if (source_->direction == io::Direction::kReversed) {
    std::swap(value->src_id, value->dst_id);
  }

  if (error::IsInvalidArgument(s) && source_->attr_info.ignore_invalid) {
    // ignore the invalid record and read next
    LOG(WARNING) << "Invalid edge data found but ignored, " << s.ToString();
    s = Read(value);
  } else if (!s.ok()) {
    LOG(WARNING) << "Invalid edge data found, " << s.ToString();
  }
  return s;
}

Status EdgeLoader::ReadRaw(Record* record) {
  Status s = reader_->Read(&record_);
  if (error::IsOutOfRange(s)) {
    LOG(INFO) << "Current edge file completed, " << source_->path;
    return s;
  } else if (!s.ok()) {
    LOG(ERROR) << "Read edge failed, " << s.ToString();
    return s;
  }
  *record = std::move(record_);
  return s;
}

Status EdgeLoader::BeginNextFile(EdgeSource** source) {
  Status s = reader_->BeginNextFile(&source_);
  if (error::IsOutOfRange(s)) {
    LOG(INFO) << "No more edge file to be read";
    return s;
  } else if (!s.ok()) {
    LOG(ERROR) << "Try to read next edge file failed, " << s.ToString();
    return s;
  }

  if (source_->src_id_type.empty() ||
      source_->dst_id_type.empty() ||
      source_->edge_type.empty()) {
    LOG(ERROR) << "Node or Edge types are not assigned, " << source_->path
               << ", src_type:" << source_->src_id_type
               << ", dst_type:" << source_->dst_id_type
               << ", edge_type:" << source_->edge_type;
    USER_LOG("Node or Edge types are not assigned.");
    return error::InvalidArgument("Node and edge types must be assigned.");
  }

  if (source) {
    *source = source_;
  }

  schema_ = reader_->GetSchema();
  return CheckSchema();
}

Status EdgeLoader::CheckSchema() {
  /// Basic schema of edge table, in which `src_id` and `dst_id` must exist.
  /// Based on the basic schema, some optional format will be added by order.
  std::vector<DataType> types = {kInt64, kInt64};
  if (source_->IsWeighted()) {
    types.push_back(kFloat);
  }
  if (source_->IsLabeled()) {
    types.push_back(kInt32);
  }
  if (source_->IsAttributed()) {
    types.push_back(kString);
  }

  RETURN_IF_ERROR(CheckSchema(types));

  ParseSideInfo(source_, &side_info_);
  side_info_.type = source_->edge_type;
  side_info_.src_type = source_->src_id_type;
  side_info_.dst_type = source_->dst_id_type;
  side_info_.direction = source_->direction;

  need_resize_ = true;

  record_.Clear();
  record_.Reserve(schema_->Size());
  return Status::OK();
}

Status EdgeLoader::CheckSchema(const std::vector<DataType>& types) {
  Schema expected(types);
  if (expected != (*schema_)) {
    std::string expected_info = expected.ToString();
    std::string actual_info = schema_->ToString();
    LOG(ERROR) << "Invalid edge source schema, " << source_->path
               << ", expect:" << expected_info
               << ", actual:" << actual_info;
    USER_LOG("The schema of edge source does not match your decoder.");
    USER_LOG(source_->path);
    return error::InvalidArgument(
      kPattern, expected_info.c_str(), actual_info.c_str());
  } else {
    return Status::OK();
  }
}

Status EdgeLoader::ParseValue(EdgeValue* value) {
  value->attrs->Clear();

  int32_t idx = 0;
  value->src_id = record_[idx++].n.l;
  value->dst_id = record_[idx++].n.l;
  if (source_->IsWeighted()) {
    value->weight = record_[idx++].n.f;
  }
  if (source_->IsLabeled()) {
    value->label = record_[idx++].n.i;
  }
  if (source_->IsAttributed()) {
    LiteString s(record_[idx].s.data, record_[idx].s.len);
    return ParseAttribute(s, source_->attr_info, value->attrs);
  }

  return Status::OK();
}

}  // namespace io
}  // namespace graphlearn
