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

#include "graphlearn/core/io/element_value.h"

#include <mutex>  //NOLINT [build/c++11]
#include <unordered_map>
#include "graphlearn/include/config.h"
#include "graphlearn/common/threading/sync/lock.h"

namespace graphlearn {
namespace io {

AttributeValue* AttributeValue::Default(const SideInfo* info) {
  static std::mutex mtx;
  static std::unordered_map<std::string, AttributeValue*> buffer;

  ScopedLocker<std::mutex> _(&mtx);
  auto it = buffer.find(info->type);
  if (it != buffer.end()) {
    return it->second;
  }

  static AttributeValue* attr = new AttributeValue;
  attr->i_attrs.assign(info->i_num, GLOBAL_FLAG(DefaultIntAttribute));
  attr->f_attrs.assign(info->f_num, GLOBAL_FLAG(DefaultFloatAttribute));
  attr->s_attrs.assign(info->s_num, GLOBAL_FLAG(DefaultStringAttribute));
  return attr;
}

}  // namespace io
}  // namespace graphlearn
