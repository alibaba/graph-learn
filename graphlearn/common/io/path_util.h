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

#ifndef GRAPHLEARN_COMMON_IO_PATH_UTIL_H_
#define GRAPHLEARN_COMMON_IO_PATH_UTIL_H_

#include <string>

namespace graphlearn {
namespace io {

std::string GetScheme(const std::string& path);
std::string GetFilePath(const std::string& path);

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_IO_PATH_UTIL_H_
