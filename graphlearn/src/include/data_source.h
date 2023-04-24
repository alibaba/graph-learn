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

#ifndef GRAPHLEARN_INCLUDE_DATA_SOURCE_H_
#define GRAPHLEARN_INCLUDE_DATA_SOURCE_H_

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
#include "include/config.h"
#include "include/constants.h"
#include "include/index_option.h"

namespace graphlearn {
namespace io {

struct AttributeInfo {
  // All of the attributes will be encoded in ONE string concatenated with
  // `delimiter`. The type of each attribute can be kInt64, kFloat and kString.
  // For kString attributes, we supply the option to convert it into kInt64,
  // which is a necessary phase when training a model in most cases. If so,
  // please make sure that the size of `hash_buckets` must be the same with
  // the size of `types`. Put `0` to `hash_buckets` for attributes that do
  // not need convertion. Otherwise, the element of `hash_buckets` is the
  // hash bucket size for each corresponding attribute.
  std::string           delimiter;
  std::vector<DataType> types;
  std::vector<int64_t>  hash_buckets;

  // Ignore the invalid attributes or not.
  bool ignore_invalid;

  AttributeInfo() : ignore_invalid(GLOBAL_FLAG(IgnoreInvalid)) {}

  AttributeInfo(const AttributeInfo& right) {
    delimiter = right.delimiter;
    types = right.types;
    hash_buckets = right.hash_buckets;
    ignore_invalid = right.ignore_invalid;
  }

  void AppendType(DataType type) {
    types.push_back(type);
  }

  void AppendHashBucket(int64_t bucket_size) {
    hash_buckets.push_back(bucket_size);
  }

  void Serialize(std::stringstream* ss) const {
    *ss << " delimiter:" << delimiter
        << " ignore_invalid:" << ignore_invalid
        << " types:";
    for (size_t i = 0; i < types.size(); ++i) {
      *ss << static_cast<int32_t>(types[i]) << ',';
    }

    *ss << " hash_buckets:";
    for (size_t i = 0; i < hash_buckets.size(); ++i) {
      *ss << static_cast<int32_t>(hash_buckets[i]) << ',';
    }
  }
};

struct NodeSource {
  std::string   path;
  std::string   id_type;
  int32_t       format;
  AttributeInfo attr_info;
  IndexOption   option;
  bool          local_shared;

  std::string view_type;
  std::string use_attrs;

  NodeSource() : format(kAttributed), local_shared(true) {
  }

  NodeSource(const NodeSource& right) {
    path = right.path;
    id_type = right.id_type;
    format = right.format;
    attr_info = right.attr_info;
    option = right.option;
    local_shared = right.local_shared;
    view_type = right.view_type;
    use_attrs = right.use_attrs;
  }

  bool IsWeighted() const {
    return format & kWeighted;
  }

  bool IsLabeled() const {
    return format & kLabeled;
  }

  bool IsTimestamped() const {
    return format & kTimestamped;
  }

  bool IsAttributed() const {
    return format & kAttributed;
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "path:" << path
       << " id_type:" << id_type
       << " view_type: " << view_type
       << " use attrs: " << use_attrs
       << " format:" << format;
    attr_info.Serialize(&ss);
    return ss.str();
  }
};

struct EdgeSource {
  std::string   path;
  std::string   edge_type;
  std::string   src_id_type;
  std::string   dst_id_type;
  int32_t       format;
  Direction     direction;
  AttributeInfo attr_info;
  IndexOption   option;
  bool          local_shared;

  std::string view_type;
  std::string use_attrs;

  EdgeSource() : format(kWeighted), direction(kOrigin), local_shared(true) {
  }

  EdgeSource(const EdgeSource& right) {
    path = right.path;
    edge_type = right.edge_type;
    src_id_type = right.src_id_type;
    dst_id_type = right.dst_id_type;
    format = right.format;
    direction = right.direction;
    attr_info = right.attr_info;
    option = right.option;
    local_shared = right.local_shared;
    view_type = right.view_type;
    use_attrs = right.use_attrs;
  }

  inline bool IsWeighted() const {
    return format & kWeighted;
  }

  inline bool IsLabeled() const {
    return format & kLabeled;
  }

  inline bool IsTimestamped() const {
    return format & kTimestamped;
  }

  inline bool IsAttributed() const {
    return format & kAttributed;
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "path:" << path
       << " edge_type:" << edge_type
       << " src_id_type:" << src_id_type
       << " dst_id_type:" << dst_id_type
       << " format:" << format
       << " view_type: " << view_type
       << " use attrs: " << use_attrs
       << " direction:" << static_cast<int32_t>(direction);
    attr_info.Serialize(&ss);
    return ss.str();
  }
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_DATA_SOURCE_H_
