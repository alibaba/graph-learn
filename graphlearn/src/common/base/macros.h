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

#ifndef GRAPHLEARN_COMMON_BASE_MACROS_H_
#define GRAPHLEARN_COMMON_BASE_MACROS_H_

#include <cstdlib>
#include "common/base/log.h"

#define ARRAYSIZE(a)                        \
  ((sizeof(a) / sizeof(*(a))) /             \
  static_cast<size_t>(!(sizeof(a) % sizeof(*(a)))))

#define BREAK_IF_NOT_OK(s)                  \
  if (!s.ok()) {                            \
    break;                                  \
  }

#define LOG_BREAK_IF_NOT_OK(s)              \
  if (!s.ok()) {                            \
    LOG(ERROR) << s.ToString();             \
    break;                                  \
  }

#define RETURN_IF_NOT_OK(s)                 \
  if (!s.ok()) {                            \
    return s;                               \
  }

#define LOG_RETURN_IF_NOT_OK(s)             \
  if (!s.ok()) {                            \
    LOG(ERROR) << s.ToString();             \
    return s;                               \
  }

#define LOG_ABORT_IF_NOT_OK(s)              \
  if (!s.ok()) {                            \
    LOG(FATAL) << s.ToString();             \
    ::exit(EXIT_FAILURE);                   \
  }

#endif  // GRAPHLEARN_COMMON_BASE_MACROS_H_
