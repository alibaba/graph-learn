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

#ifndef GRAPHLEARN_CORE_IO_PARSER_H_
#define GRAPHLEARN_CORE_IO_PARSER_H_

#include <string>
#include <vector>
#include "graphlearn/common/string/lite_string.h"
#include "graphlearn/core/io/element_value.h"
#include "graphlearn/include/data_source.h"
#include "graphlearn/include/status.h"

namespace graphlearn {
namespace io {

Status ParseAttribute(
    const LiteString& input,
    const AttributeInfo& info,
    AttributeValue* value);

template<class T>
void ParseSideInfo(const T* source, SideInfo* info) {
  info->i_num = 0;
  info->f_num = 0;
  info->s_num = 0;
  info->format = source->format;

  const AttributeInfo& attr_info = source->attr_info;
  for (int32_t i = 0; i < attr_info.types.size(); ++i) {
    if (attr_info.types[i] == DataType::kInt32 ||
        attr_info.types[i] == DataType::kInt64) {
      ++(info->i_num);
    } else if (attr_info.types[i] == DataType::kFloat ||
               attr_info.types[i] == DataType::kDouble) {
      ++(info->f_num);
    } else {
      if (attr_info.hash_buckets.empty()) {
        ++(info->s_num);
      } else if (attr_info.hash_buckets[i] > 0) {
        // hash string to int, so we treat it as an integer
        ++(info->i_num);
      } else {
        ++(info->s_num);
      }
    }
  }
}

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_IO_PARSER_H_
