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

#include "loader.h"

#include <fstream>

namespace dgs {
namespace dataloader {
namespace file {

void FileLoader::Load(const std::string& file_path) {
  std::ifstream infile;
  infile.open(file_path);
  if (!infile.good()) {
    throw std::runtime_error("cannot open file: " + file_path);
  }
  auto& line_processor = LineProcessor::GetInstance();
  std::string line;
  while (std::getline(infile, line)) {
    line_processor.Process(line, group_producer_);
  }
  group_producer_.FlushAll();
  infile.close();
}

}  // namespace file
}  // namespace dataloader
}  // namespace dgs
