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

#include "graphlearn/core/io/parser.h"

#include <utility>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/hash.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/common/string/numeric.h"
#include "graphlearn/common/string/string_tool.h"

namespace graphlearn {
namespace io {

namespace {

int64_t Hash(const std::string& data, int64_t bucket) {
  uint64_t h = ::graphlearn::Hash64(data);
  return h % bucket;
}

const char* pattern = "The %dth attribute expect an %s, but got \"%s\".";

}  // anonymous namespace

Status ParseAttribute(const LiteString& input,
                      const AttributeInfo& info,
                      AttributeValue* value) {
  const std::string& delimiter = info.delimiter;
  const std::vector<DataType>& types = info.types;
  const std::vector<int64_t>& hash_buckets = info.hash_buckets;

  std::vector<std::string> attrs =
    ::graphlearn::strings::Split(input, delimiter);

  if (attrs.size() != types.size()) {
    LOG(ERROR) << "The count of attributes does not match your decoder"
               << ", expect:" << types.size()
               << ", actual:" << attrs.size();
    return error::InvalidArgument("Unexpected attribute count");
  }

  for (size_t i = 0; i < attrs.size(); ++i) {
    DataType type = types[i];
    if (type == DataType::kInt32) {
      int32_t v = 0;
      if (!::graphlearn::strings::FastStringTo32(attrs[i].c_str(), &v)) {
        LOG(ERROR) << "Invalid attribute:" << attrs[i] << "\t" << i;
        return error::InvalidArgument(pattern, i, "int", attrs[i].c_str());
      }
      value->Add(static_cast<int64_t>(v));
    } else if (type == DataType::kInt64) {
      int64_t v = 0;
      if (!::graphlearn::strings::FastStringTo64(attrs[i].c_str(), &v)) {
        LOG(ERROR) << "Invalid attribute:" << attrs[i] << "\t" << i;
        return error::InvalidArgument(pattern, i, "int64", attrs[i].c_str());
      }
      value->Add(v);
    } else if (type == DataType::kFloat) {
      float v = 0.0;
      if (!::graphlearn::strings::FastStringToFloat(attrs[i].c_str(), &v)) {
        LOG(ERROR) << "Invalid attribute:" << attrs[i] << "\t" << i;
        return error::InvalidArgument(pattern, i, "float", attrs[i].c_str());
      }
      value->Add(v);
    } else if (type == DataType::kDouble) {
      double v = 0.0;
      if (!::graphlearn::strings::FastStringToDouble(attrs[i].c_str(), &v)) {
        LOG(ERROR) << "Invalid attribute:" << attrs[i] << "\t" << i;
        return error::InvalidArgument(pattern, i, "double", attrs[i].c_str());
      }
      value->Add(static_cast<float>(v));
    } else if (type == DataType::kString) {
      if (hash_buckets.empty()) {
        value->Add(std::move(attrs[i]));
      } else if (hash_buckets[i] > 0) {
        value->Add(Hash(attrs[i], hash_buckets[i]));
      } else {
        value->Add(std::move(attrs[i]));
      }
    } else {
      LOG(WARNING) << "Could not reach here";
    }
  }
  return Status::OK();
}

}  // namespace io
}  // namespace graphlearn

