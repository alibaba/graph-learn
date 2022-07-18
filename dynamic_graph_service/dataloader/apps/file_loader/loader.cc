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

#include <iostream>

#include "dataloader/utils.h"

namespace dgs {
namespace dataloader {
namespace file {

char delimiter = '&';
uint32_t batch_size = 16;

FileLoader::FileLoader(const std::string& pattern_file)
  : group_producer_(batch_size) {
  std::ifstream infile;
  infile.open(pattern_file);
  if (!infile.good()) {
    throw std::runtime_error("cannot open pattern file: " + pattern_file);
  }
  std::string line;
  while (std::getline(infile, line)) {
    auto patterns = StrSplit(line, delimiter);
    if (StartsWith(patterns[0], "#VERTEX")) {
      AddVertexPattern(std::move(patterns));
    } else if (StartsWith(patterns[0], "#EDGE")) {
      AddEdgePattern(std::move(patterns));
    } else {
      std::cerr << "Unsupported pattern type: " << patterns[0] << ", ignore!" << std::endl;
    }
  }
  infile.close();
}

void FileLoader::Load(const std::string& file_path) {
  std::ifstream infile;
  infile.open(file_path);
  if (!infile.good()) {
    throw std::runtime_error("cannot open file: " + file_path);
  }
  std::string line;
  while (std::getline(infile, line)) {
    ProcessLine(line);
  }
  group_producer_.FlushAll();
  infile.close();
}

std::vector<FileLoader::AttrParseFunc> FileLoader::GetAttrParsers(std::string* attr_pattern, size_t n) {
  std::vector<AttrParseFunc> attr_parsers;
  attr_parsers.reserve(n);
  auto& schema = Schema::Get();
  for (size_t i = 0; i < n; i++) {
    auto& attr_def = schema.GetAttrDefByName(attr_pattern[i]);
    auto attr_type = attr_def.Type();
    auto attr_value_type = attr_def.ValueType();
    switch (attr_value_type) {
      case BOOL: {
        std::cerr << "attribute bool type not implemented" << std::endl;
        break;
      }
      case CHAR: {
        std::cerr << "attribute char type not implemented" << std::endl;
        break;
      }
      case INT16: {
        std::cerr << "attribute int16 type not implemented" << std::endl;
        break;
      }
      case INT32: {
        AttrParseFunc func = [attr_type, attr_value_type] (std::string&& attr_str) {
          int32_t attr = std::stoi(attr_str);
          std::string attr_bytes(reinterpret_cast<const char*>(&attr), sizeof(attr));
          return AttrInfo{attr_type, attr_value_type, std::move(attr_bytes)};
        };
        attr_parsers.emplace_back(std::move(func));
        break;
      }
      case INT64: {
        AttrParseFunc func = [attr_type, attr_value_type] (std::string&& attr_str) {
          int64_t attr = std::stoll(attr_str);
          std::string attr_bytes(reinterpret_cast<const char*>(&attr), sizeof(attr));
          return AttrInfo{attr_type, attr_value_type, std::move(attr_bytes)};
        };
        attr_parsers.emplace_back(std::move(func));
        break;
      }
      case FLOAT32: {
        AttrParseFunc func = [attr_type, attr_value_type] (std::string&& attr_str) {
          float attr = std::stof(attr_str);
          std::string attr_bytes(reinterpret_cast<const char*>(&attr), sizeof(attr));
          return AttrInfo{attr_type, attr_value_type, std::move(attr_bytes)};
        };
        attr_parsers.emplace_back(std::move(func));
        break;
      }
      case FLOAT64: {
        AttrParseFunc func = [attr_type, attr_value_type] (std::string&& attr_str) {
          double attr = std::stod(attr_str);
          std::string attr_bytes(reinterpret_cast<const char*>(&attr), sizeof(attr));
          return AttrInfo{attr_type, attr_value_type, std::move(attr_bytes)};
        };
        attr_parsers.emplace_back(std::move(func));
        break;
      }
      case STRING: {
        AttrParseFunc func = [attr_type, attr_value_type] (std::string&& attr_str) {
          return AttrInfo{attr_type, attr_value_type, std::move(attr_str)};
        };
        attr_parsers.emplace_back(std::move(func));
        break;
      }
      case BYTES: {
        std::cerr << "attribute bytes type not implemented" << std::endl;
        break;
      }
      default:
        std::cerr << "Unsupported attribute value type of " << attr_pattern[i] << std::endl;
        break;
    }
  }
  return attr_parsers;
}

void FileLoader::AddVertexPattern(std::vector<std::string>&& line_patterns) {
  if (line_patterns.size() < 2) {
    std::cerr << "Incorrect filed number of vertex pattern type: " << line_patterns[0] << ", ignore!" << std::endl;
    return;
  }
  auto vname = std::move(StrSplit(line_patterns[0], ':')[1]);
  auto vtype = Schema::Get().GetVertexDefByName(vname).Type();
  auto attr_parsers = GetAttrParsers(line_patterns.data() + 2, line_patterns.size() - 2);
  LineProcessFunc func = [vtype, attr_parsers = std::move(attr_parsers)] (std::string* line_pattern, size_t n, GroupProducer& p) {
    if (n != (2 + attr_parsers.size())) {
      std::cerr << "Load invalid vertex record, ignore!" << std::endl;
      return;
    }
    VertexId vid = std::stoll(line_pattern[1]);
    std::string* attr_pattern = line_pattern + 2;
    size_t attr_num = n - 2;
    std::vector<AttrInfo> attr_infos;
    attr_infos.reserve(attr_num);
    for (size_t i = 0; i < attr_num; i++) {
      attr_infos.emplace_back(attr_parsers[i](std::move(attr_pattern[i])));
    }
    p.AddVertex(vtype, vid, attr_infos);
  };
  processors_.emplace(vname, std::move(func));
}

void FileLoader::AddEdgePattern(std::vector<std::string>&& line_patterns) {
  if (line_patterns.size() < 3) {
    std::cerr << "Incorrect filed number of edge pattern type: " << line_patterns[0] << ", ignore!" << std::endl;
    return;
  }
  if (!StartsWith(line_patterns[1], "#SRC")) {
    std::cerr << "Invalid filed of src type of edge pattern type: " << line_patterns[0] << ", ignore!" << std::endl;
    return;
  }
  if (!StartsWith(line_patterns[2], "#DST")) {
    std::cerr << "Invalid filed of dst type of edge pattern type: " << line_patterns[0] << ", ignore!" << std::endl;
    return;
  }
  auto& schema = Schema::Get();
  auto ename = std::move(StrSplit(line_patterns[0], ':')[1]);
  auto etype = schema.GetEdgeDefByName(ename).Type();
  auto src_vname = std::move(StrSplit(line_patterns[1], ':')[1]);
  auto src_vtype = schema.GetVertexDefByName(src_vname).Type();
  auto dst_vname = std::move(StrSplit(line_patterns[2], ':')[1]);
  auto dst_vtype = schema.GetVertexDefByName(dst_vname).Type();
  auto attr_parsers = GetAttrParsers(line_patterns.data() + 3, line_patterns.size() - 3);
  LineProcessFunc func = [etype, src_vtype, dst_vtype, attr_parsers = std::move(attr_parsers)]
      (std::string* line_pattern, size_t n, GroupProducer& p) {
    if (n != (3 + attr_parsers.size())) {
      std::cerr << "Load invalid edge record, ignore!" << std::endl;
      return;
    }
    VertexId src_vid = std::stoll(line_pattern[1]);
    VertexId dst_vid = std::stoll(line_pattern[2]);
    std::string* attr_pattern = line_pattern + 3;
    size_t attr_num = n - 3;
    std::vector<AttrInfo> attr_infos;
    attr_infos.reserve(attr_num);
    for (size_t i = 0; i < attr_num; i++) {
      attr_infos.emplace_back(attr_parsers[i](std::move(attr_pattern[i])));
    }
    p.AddEdge(etype, src_vtype, dst_vtype, src_vid, dst_vid, attr_infos);
  };
  processors_.emplace(ename, std::move(func));
}

void FileLoader::ProcessLine(const std::string& line) {
  auto records = StrSplit(line, delimiter);
  auto& func = processors_.at(records[0]);
  func(records.data(), records.size(), group_producer_);
}

}  // namespace file
}  // namespace dataloader
}  // namespace dgs
