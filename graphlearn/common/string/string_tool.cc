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

#include "graphlearn/common/string/string_tool.h"

#include <cctype>
#include <vector>

namespace graphlearn {
namespace strings {

namespace {

struct AllowAll {
  bool operator()(LiteString sp) const {
    return true;
  }
};

template <typename Filter>
std::vector<std::string> Split(LiteString text, LiteString delims,
                               Filter f) {
  std::vector<std::string> result;
  size_t token_start = 0;
  if (!text.empty()) {
    for (size_t i = 0; i < text.size() + 1; i++) {
      if ((i == text.size()) || (delims.find(text[i]) != LiteString::npos)) {
        LiteString token(text.data() + token_start, i - token_start);
        if (f(token)) {
          result.push_back(token.ToString());
        }
        token_start = i + 1;
      }
    }
  }
  return result;
}

}  // anonymous namespace

std::string Lowercase(LiteString s) {
  std::string result(s.data(), s.size());
  for (char& c : result) {
    c = ::tolower(c);
  }
  return result;
}

std::string Uppercase(LiteString s) {
  std::string result(s.data(), s.size());
  for (char& c : result) {
    c = ::toupper(c);
  }
  return result;
}

std::vector<std::string> Split(LiteString text, LiteString delims) {
  return Split(text, delims, AllowAll());
}

std::vector<std::string> Split(LiteString text, char delim) {
  return Split(text, LiteString(&delim, 1));
}

void StripHead(std::string* s) {
  std::string::size_type i = 0;
  for (; i < s->size() && ::isspace((*s)[i]); ++i) {
  }
  s->erase(0, i);
}

void StripTail(std::string* s) {
  std::string::size_type i = s->size();
  for (; i > 0 && ::isspace((*s)[i - 1]); --i) {
  }
  s->resize(i);
}

void StripContext(std::string* s) {
  StripTail(s);
  StripHead(s);
}

size_t StripHead(LiteString* text) {
  size_t count = 0;
  const char* ptr = text->data();
  while (count < text->size() && ::isspace(*ptr)) {
    count++;
    ptr++;
  }
  text->remove_prefix(count);
  return count;
}

size_t StripTail(LiteString* text) {
  size_t count = 0;
  const char* ptr = text->data() + text->size() - 1;
  while (count < text->size() && ::isspace(*ptr)) {
    ++count;
    --ptr;
  }
  text->remove_suffix(count);
  return count;
}

size_t StripContext(LiteString* text) {
  return (StripHead(text) + StripTail(text));
}

bool ConsumePrefix(LiteString* s, LiteString expected) {
  if (s->starts_with(expected)) {
    s->remove_prefix(expected.size());
    return true;
  }
  return false;
}

bool ConsumeSuffix(LiteString* s, LiteString expected) {
  if (s->ends_with(expected)) {
    s->remove_suffix(expected.size());
    return true;
  }
  return false;
}

bool StartWith(const std::string& s, const std::string& pattern) {
  if (s.size() < pattern.size()) {
    return false;
  }
  for (size_t i = 0; i < pattern.size(); ++i) {
    if (s[i] != pattern[i]) {
      return false;
    }
  }
  return true;
}

bool EndWith(const std::string& s, const std::string& pattern) {
  if (s.size() < pattern.size()) {
    return false;
  }
  const char* cs = &(s.back());
  const char* cp = &(pattern.back());
  for (size_t i = 0; i < pattern.size(); ++i, --cs, --cp) {
    if ((*cs) != *(cp)) {
      return false;
    }
  }
  return true;
}

}  // namespace strings
}  // namespace graphlearn
