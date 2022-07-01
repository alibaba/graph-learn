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

#include "graphlearn/common/io/path_util.h"

namespace graphlearn {
namespace io {

std::string GetScheme(const std::string& path) {
  std::string::size_type pos = path.find("://");
  if (pos == std::string::npos) {
    return "";
  }
  return path.substr(0, pos);
}

std::string GetFilePath(const std::string& path) {
  std::string::size_type pos = path.find("://");
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 3);
}

void ParseURI(const std::string& uri, std::string* scheme,
              std::string* host, std::string* path) {
  std::string remain = uri;
  std::string::size_type pos = remain.find("://");
  if (pos == std::string::npos) {
    *path = remain;
    return;
  }
  *scheme = remain.substr(0, pos);
  remain = remain.substr(pos + 3);
  pos = remain.find("/");
  if (pos == std::string::npos) {
    *host = remain;
    return;
  }
  *host = remain.substr(0, pos);
  *path = remain.substr(pos);
}

std::string BaseName(const std::string& uri) {
  std::string scheme, host, path;
  ParseURI(uri, &scheme, &host, &path);
  std::string::size_type pos = path.rfind("/");
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

}  // namespace io
}  // namespace graphlearn
