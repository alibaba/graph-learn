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

#ifndef GRAPHLEARN_CORE_IO_EDGE_LOADER_H_
#define GRAPHLEARN_CORE_IO_EDGE_LOADER_H_

#include <memory>
#include <vector>
#include "graphlearn/common/io/value.h"
#include "graphlearn/core/io/element_value.h"
#include "graphlearn/include/data_source.h"
#include "graphlearn/include/status.h"

namespace graphlearn {

class Env;

namespace io {

template <typename SourceType>
class SliceReader;

class EdgeLoader {
public:
  EdgeLoader(const std::vector<EdgeSource>& source,
             Env* env,
             int32_t thread_id,
             int32_t thread_num);
  ~EdgeLoader();

  Status BeginNextFile();
  Status Read(EdgeValue* value);

  const SideInfo* GetSideInfo() const {
    return &side_info_;
  }

private:
  Status CheckSchema();
  Status CheckSchema(const std::vector<DataType>& types);
  Status ParseValue(EdgeValue *value);

private:
  SliceReader<EdgeSource>* reader_;
  EdgeSource*              source_;
  Record                   record_;
  const Schema*            schema_;
  SideInfo                 side_info_;
  bool                     need_resize_;
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_IO_EDGE_LOADER_H_
