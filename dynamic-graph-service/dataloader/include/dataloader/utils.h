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

#ifndef DATALOADER_UTILS_H_
#define DATALOADER_UTILS_H_

#include <string>
#include <vector>

namespace dgs {
namespace dataloader {

inline
std::string StrJoin(const std::vector<std::string>& lists, const std::string& delim) {
  if (lists.empty()) {
    return "";
  }
  std::string join = lists[0];
  for (size_t i = 1; i < lists.size(); i++) {
    join += delim;
    join += lists[i];
  }
  return join;
}

inline
std::vector<std::string> StrSplit(const std::string& str, char delim) {
  std::vector<std::string> tokens;
  size_t prev = 0, pos;
  do {
    pos = str.find(delim, prev);
    if (pos == std::string::npos) {
      pos = str.length();
    }
    tokens.emplace_back(str.substr(prev, pos - prev));
    prev = pos + 1;
  } while (pos < str.length() && prev < str.length());
  return tokens;
}

inline
bool StartsWith(const std::string& str, const std::string& match) {
  return (str.rfind(match, 0) == 0);
}

}  // namespace dataloader
}  // namespace dgs

#endif // DATALOADER_UTILS_H_
