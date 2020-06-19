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

#include "graphlearn/core/io/node_loader.h"

#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/core/io/parser.h"
#include "graphlearn/core/io/slice_reader.h"
#include "graphlearn/platform/env.h"

namespace graphlearn {
namespace io {

namespace {

const char* kPattern = "Invalid node table schema, expected: %s, but got: %s";

}  // anonymous namespace

NodeLoader::NodeLoader(const std::vector<NodeSource>& source,
                       Env* env,
                       int32_t thread_id,
                       int32_t thread_num)
    : source_(nullptr),
      need_resize_(false) {
  reader_ = new SliceReader<NodeSource>(
    source, env, thread_id, thread_num);
}

NodeLoader::~NodeLoader() {
  delete reader_;
}

Status NodeLoader::Read(NodeValue* value) {
  Status s = reader_->Read(&record_);
  if (error::IsOutOfRange(s)) {
    LOG(INFO) << "Current node file completed, " << source_->path;
    return s;
  } else if (!s.ok()) {
    LOG(ERROR) << "Read node failed, " << s.ToString();
    return s;
  }

  if (need_resize_) {
    value->Reserve(side_info_.i_num,
                   side_info_.f_num,
                   side_info_.s_num);
    need_resize_ = false;
  }

  s = ParseValue(value);
  if (error::IsInvalidArgument(s) && source_->ignore_invalid) {
    // ignore the invalid record and read next
    LOG(WARNING) << "Invalid node data found but ignored, " << s.ToString();
    s = Read(value);
  } else if (!s.ok()) {
    LOG(WARNING) << "Invalid node data found, " << s.ToString();
  }
  return s;
}

Status NodeLoader::BeginNextFile() {
  Status s = reader_->BeginNextFile(&source_);
  if (error::IsOutOfRange(s)) {
    LOG(INFO) << "No more node file to be read";
    return s;
  } else if (!s.ok()) {
    LOG(ERROR) << "Try to read next node file failed, " << s.ToString();
    return s;
  }

  if (source_->id_type.empty()) {
    LOG(ERROR) << "Node type is not assigned, " << source_->path;
    USER_LOG("Node type is not assigned.");
    return error::InvalidArgument("Node id type must be assigned.");
  }

  schema_ = reader_->GetSchema();
  return CheckSchema();
}

Status NodeLoader::CheckSchema() {
  std::vector<DataType> types;

  // node id
  types.push_back(kInt64);

  if (source_->IsWeighted()) {
    types.push_back(kFloat);
  }
  if (source_->IsLabeled()) {
    types.push_back(kInt32);
  }
  if (source_->IsAttributed()) {
    types.push_back(kString);
  }

  RETURN_IF_ERROR(CheckTableSchema(types));

  ParseSideInfo(source_, &side_info_);
  side_info_.type = source_->id_type;

  need_resize_ = true;

  record_.Clear();
  record_.Reserve(schema_->Size());
  return Status::OK();
}

Status NodeLoader::CheckTableSchema(const std::vector<DataType>& types) {
  Schema expected(types);
  if (expected != (*schema_)) {
    std::string expected_info = expected.ToString();
    std::string actual_info = schema_->ToString();
    LOG(ERROR) << "Invalid node source schema, " << source_->path
               << ", expect:" << expected_info
               << ", actual:" << actual_info;
    USER_LOG("The schema of node source does not match your decoder.");
    USER_LOG(source_->path);
    return error::InvalidArgument(
      kPattern, expected_info.c_str(), actual_info.c_str());
  } else {
    return Status::OK();
  }
}

Status NodeLoader::ParseValue(NodeValue* value) {
  value->Clear();

  int idx = 0;
  value->id = record_[idx++].n.l;

  if (source_->IsWeighted()) {
    value->weight = record_[idx++].n.f;
  }
  if (source_->IsLabeled()) {
    value->label = record_[idx++].n.i;
  }
  if (source_->IsAttributed()) {
    LiteString s(record_[idx].s.data, record_[idx].s.len);
    return ParseAttribute(s, source_->delimiter,
                          source_->types, source_->hash_buckets,
                          value);
  }

  return Status::OK();
}

}  // namespace io
}  // namespace graphlearn
