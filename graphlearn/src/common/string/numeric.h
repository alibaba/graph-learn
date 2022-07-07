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

#ifndef GRAPHLEARN_COMMON_STRING_NUMERIC_H_
#define GRAPHLEARN_COMMON_STRING_NUMERIC_H_

#include <string>
#include "common/string/lite_string.h"

namespace graphlearn {
namespace strings {

// Leading and trailing spaces are allowed.
// Return false with overflow or invalid input.
bool SafeStringTo32(LiteString str, int32_t* value);
bool SafeStringTo64(LiteString str, int64_t* value);

bool FastStringTo32(const char* str, int32_t* value);
bool FastStringTo64(const char* str, int64_t* value);
bool FastStringToFloat(const char* str, float* value);
bool FastStringToDouble(const char* str, double* value);

std::string Int32ToString(int32_t i);
std::string UInt32ToString(uint32_t u);
std::string Int64ToString(int64_t i);
std::string UInt64ToString(uint64_t u);

}  // namespace strings
}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_STRING_NUMERIC_H_
