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

#ifndef GRAPHLEARN_COMMON_STRING_STRING_TOOL_H_
#define GRAPHLEARN_COMMON_STRING_STRING_TOOL_H_

#include <limits.h>
#include <functional>
#include <string>
#include <vector>
#include "graphlearn/common/string/lite_string.h"

namespace graphlearn {
namespace strings {

std::string Lowercase(LiteString s);
std::string Uppercase(LiteString s);

// Split strings using any of the supplied delimiters. For example:
// Split("a,b.c,d", ".,") would return {"a", "b", "c", "d"}.
std::vector<std::string> Split(LiteString text, LiteString delims);

// Split strings using the supplied delimiter.
std::vector<std::string> Split(LiteString text, char delim);

// Removes whitespace from "*s".
void StripHead(std::string* s);
void StripTail(std::string* s);
void StripContext(std::string* s);

// Removes whitespace from text and returns number of characters removed.
size_t StripHead(LiteString* text);
size_t StripTail(LiteString* text);
size_t StripContext(LiteString* text);

bool StartWith(const std::string& s, const std::string& pattern);
bool EndWith(const std::string& s, const std::string& pattern);

std::string Join(const std::vector<std::string>& v,
                 LiteString delim,
                 uint32_t from = 0,
                 uint32_t to = UINT_MAX);

}  // namespace strings
}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_STRING_STRING_TOOL_H_
