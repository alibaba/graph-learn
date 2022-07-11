/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef FILE_LOADER_LINE_PROCESSOR_H_
#define FILE_LOADER_LINE_PROCESSOR_H_

#include "dataloader/schema.h"
#include "dataloader/typedefs.h"

#include "group_producer.h"

namespace dgs {
namespace dataloader {
namespace file {

extern char delimiter;

class LineProcessor {
  using LineProcessFunc = std::function<void(std::string*, size_t, GroupProducer&)>;
  using AttrParseFunc = std::function<AttrInfo(std::string&&)>;
public:
  static LineProcessor& GetInstance() {
    static LineProcessor instance;
    return instance;
  }

  void Init(const std::string& pattern_file);

  void Process(const std::string& line, GroupProducer& p);

private:
  LineProcessor() = default;

  void AddVertexPattern(std::vector<std::string>&& line_patterns);
  void AddEdgePattern(std::vector<std::string>&& line_patterns);

  static std::vector<AttrParseFunc> GetAttrParsers(std::string* attr_pattern, size_t n);

private:
  std::unordered_map<std::string, LineProcessFunc> processors_;
};

}  // namespace file
}  // namespace dataloader
}  // namespace dgs

#endif // FILE_LOADER_LINE_PROCESSOR_H_
