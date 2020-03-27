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

#include "graphlearn/common/string/numeric.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <locale>
#include <unordered_map>
#include "graphlearn/common/base/macros.h"

namespace graphlearn {
namespace strings {

namespace {

const int16_t kInt16Min = ((int16_t)~0x7FFF);
const int16_t kInt16Max = ((int16_t)0x7FFF);
const int32_t kInt32Min = ((int32_t)~0x7FFFFFFF);
const int32_t kInt32Max = ((int32_t)0x7FFFFFFF);
const int64_t kInt64Min = ((int64_t)~0x7FFFFFFFFFFFFFFFll);
const int64_t kInt64Max = ((int64_t)0x7FFFFFFFFFFFFFFFll);

char* FastUInt32ToBuffer(uint32_t i, char* buffer) {
  char* start = buffer;
  do {
    *buffer++ = ((i % 10) + '0');
    i /= 10;
  } while (i > 0);
  *buffer = '\0';
  std::reverse(start, buffer);
  return buffer;
}

char* FastInt32ToBuffer(int32_t i, char* buffer) {
  uint32_t u = i;
  if (i < 0) {
    *buffer++ = '-';
    u = 0 - u;
  }
  return FastUInt32ToBuffer(u, buffer);
}

char* FastUInt64ToBuffer(uint64_t i, char* buffer) {
  char* start = buffer;
  do {
    *buffer++ = ((i % 10) + '0');
    i /= 10;
  } while (i > 0);
  *buffer = '\0';
  std::reverse(start, buffer);
  return buffer;
}

char* FastInt64ToBuffer(int64_t i, char* buffer) {
  uint64_t u = i;
  if (i < 0) {
    *buffer++ = '-';
    u = 0 - u;
  }
  return FastUInt64ToBuffer(u, buffer);
}

char SafeFirstChar(LiteString str) {
  if (str.empty()) {
    return '\0';
  }
  return str[0];
}

void SkipSpaces(LiteString* str) {
  while (isspace(SafeFirstChar(*str))) {
    str->remove_prefix(1);
  }
}

}  // anonymous namespace

bool SafeStringTo32(LiteString str, int32_t* value) {
  SkipSpaces(&str);

  int64_t vmax = kInt32Max;
  int sign = 1;
  if (str.Consume("-")) {
    sign = -1;
    ++vmax;
  }

  if (!isdigit(SafeFirstChar(str))) {
    return false;
  }

  int64_t result = 0;
  do {
    result = result * 10 + SafeFirstChar(str) - '0';
    if (result > vmax) {
      return false;
    }
    str.remove_prefix(1);
  } while (isdigit(SafeFirstChar(str)));

  SkipSpaces(&str);

  if (!str.empty()) {
    return false;
  }

  *value = static_cast<int32_t>(result * sign);
  return true;
}

bool SafeStringTo64(LiteString str, int64_t* value) {
  SkipSpaces(&str);

  int64_t vlimit = kInt64Max;
  int sign = 1;
  if (str.Consume("-")) {
    sign = -1;
    vlimit = kInt64Min;
  }

  if (!isdigit(SafeFirstChar(str))) {
    return false;
  }

  int64_t result = 0;
  if (sign == 1) {
    do {
      int digit = SafeFirstChar(str) - '0';
      if ((vlimit - digit) / 10 < result) {
        return false;
      }
      result = result * 10 + digit;
      str.remove_prefix(1);
    } while (isdigit(SafeFirstChar(str)));
  } else {
    do {
      int digit = SafeFirstChar(str) - '0';
      if ((vlimit + digit) / 10 > result) {
        return false;
      }
      result = result * 10 - digit;
      str.remove_prefix(1);
    } while (isdigit(SafeFirstChar(str)));
  }

  SkipSpaces(&str);
  if (!str.empty()) {
    return false;
  }

  *value = result;
  return true;
}

bool FastStringTo32(const char* str, int32_t* value) {
  char* end = nullptr;
  errno = 0;
  int64_t ret = strtol(str, &end, 10);
  while (isspace(*end)) ++end;
  if (*end == '\0' && errno == 0 &&
      ret <= std::numeric_limits<int>::max() &&
      ret >= std::numeric_limits<int>::min() ) {
    *value = ret;
    return true;
  }
  return false;
}

bool FastStringTo64(const char* str, int64_t* value) {
  char* end = nullptr;
  int64_t ret = strtol(str, &end, 10);
  while (isspace(*end)) ++end;
  if (*end == '\0') {
    *value = ret;
    return true;
  }
  return false;
}

bool FastStringToFloat(const char* str, float* value) {
  char* end = nullptr;
  float ret = strtof(str, &end);
  while (isspace(*end)) ++end;
  if (*end == '\0') {
    *value = ret;
    return true;
  }
  return false;
}

bool FastStringToDouble(const char* str, double* value) {
  char* end = nullptr;
  double ret = strtod(str, &end);
  while (isspace(*end)) ++end;
  if (*end == '\0') {
    *value = ret;
    return true;
  }
  return false;
}

std::string Int32ToString(int32_t i) {
  char buffer[12];
  FastInt32ToBuffer(i, buffer);
  return std::string(buffer);
}

std::string UInt32ToString(uint32_t u) {
  char buffer[12];
  FastUInt32ToBuffer(u, buffer);
  return std::string(buffer);
}

std::string Int64ToString(int64_t i) {
  char buffer[24];
  FastInt64ToBuffer(i, buffer);
  return std::string(buffer);
}

std::string UInt64ToString(uint64_t u) {
  char buffer[24];
  FastUInt64ToBuffer(u, buffer);
  return std::string(buffer);
}

}  // namespace strings
}  // namespace graphlearn
