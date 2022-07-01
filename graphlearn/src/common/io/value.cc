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

#include "graphlearn/common/io/value.h"

namespace graphlearn {
namespace io {

DataType ToDataType(const std::string& type) {
  if (type == "int" || type == "int32") {
    return kInt32;
  } else if (type == "long" || type == "int64") {
    return kInt64;
  } else if (type == "float") {
    return kFloat;
  } else if (type == "double") {
    return kDouble;
  } else if (type == "string") {
    return kString;
  }

  return kUnknown;
}

}  // namespace io
}  // namespace graphlearn
