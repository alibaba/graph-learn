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
#include "graphlearn/include/constants.h"

namespace graphlearn {
namespace io {

struct NodeSource {
  std::string path;
  std::string id_type;
  int32_t     format;
  bool        ignore_invalid;

  // attribute delimiter and type list
  std::string delimiter;
  std::vector<DataType> types;

  // whether to hash string attributes into integer.
  // Default empty. Otherwise, must be the same size with types.
  // 0 means no need hash.
  std::vector<int64_t> hash_buckets;

  NodeSource()
      : format(kAttributed),
        ignore_invalid(false) {
  }

  NodeSource(const NodeSource& right) {
    path = right.path;
    id_type = right.id_type;
    format = right.format;
    ignore_invalid = right.ignore_invalid;
    delimiter = right.delimiter;
    types = right.types;
    hash_buckets = right.hash_buckets;
  }

  void AppendAttrType(DataType type) {
    types.push_back(type);
  }

  void AppendHashBucket(int64_t bucket_size) {
    hash_buckets.push_back(bucket_size);
  }

  bool IsWeighted() const {
    return format & kWeighted;
  }

  bool IsLabeled() const {
    return format & kLabeled;
  }

  bool IsAttributed() const {
    return format & kAttributed;
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "path:" << path
       << " id_type:" << id_type
       << " format:" << format
       << " delimiter:" << delimiter
       << " ignore_invalid:" << ignore_invalid;
    ss << " types:";
    for (size_t i = 0; i < types.size(); ++i) {
      ss << static_cast<int32_t>(types[i]) << ',';
    }
    ss << " hash_buckets:";
    for (size_t i = 0; i < hash_buckets.size(); ++i) {
      ss << static_cast<int32_t>(hash_buckets[i]) << ',';
    }

    return ss.str();
  }
};

struct EdgeSource {
  std::string path;
  std::string edge_type;
  std::string src_id_type;
  std::string dst_id_type;
  int32_t     format;
  bool        ignore_invalid;
  Direction   direction;

  // attribute delimiter and type list
  std::string delimiter;
  std::vector<DataType> types;

  // whether to hash string attributes into integer.
  // Default empty. Otherwise, must be the same size with types.
  // 0 means no need hash.
  std::vector<int64_t> hash_buckets;

  EdgeSource()
      : format(kWeighted),
        direction(kOrigin) {
  }

  EdgeSource(const EdgeSource& right) {
    path = right.path;
    edge_type = right.edge_type;
    src_id_type = right.src_id_type;
    dst_id_type = right.dst_id_type;
    format = right.format;
    ignore_invalid = right.ignore_invalid;
    direction = right.direction;
    delimiter = right.delimiter;
    types = right.types;
    hash_buckets = right.hash_buckets;
  }

  inline void AppendAttrType(DataType type) {
    types.push_back(type);
  }

  inline void AppendHashBucket(int64_t bucket_size) {
    hash_buckets.push_back(bucket_size);
  }

  inline bool IsWeighted() const {
    return format & kWeighted;
  }

  inline bool IsLabeled() const {
    return format & kLabeled;
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
       << " ignore_invalid:" << ignore_invalid
       << " direction:" << static_cast<int32_t>(direction)
       << " delimiter:" << delimiter
       << " attr_types:";
    for (size_t i = 0; i < types.size(); ++i) {
      ss << static_cast<int32_t>(types[i]) << ',';
    }
    ss << " hash_buckets:";
    for (size_t i = 0; i < hash_buckets.size(); ++i) {
      ss << static_cast<int32_t>(hash_buckets[i]) << ',';
    }
    return ss.str();
  }
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_DATA_SOURCE_H_
