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

#ifndef GRAPHLEARN_COMMON_BASE_ERRORS_H_
#define GRAPHLEARN_COMMON_BASE_ERRORS_H_

#include <string>
#include <vector>
#include "include/status.h"

namespace graphlearn {
namespace error {

#define RETURN_IF_ERROR(...)                      \
  do {                                            \
    const ::graphlearn::Status _ = (__VA_ARGS__); \
    if (!_.ok()) return _;                        \
  } while (0)


#define DECLARE_ERROR(Func, Type)                            \
  extern ::graphlearn::Status Func(const std::string& msg);  \
  inline bool Is##Func(const ::graphlearn::Status& status) { \
    return status.code() == ::graphlearn::error::Type;       \
  }                                                          \
  template<typename ... T>                                   \
  ::graphlearn::Status Func(T... args) {                     \
    char msg[128];                                           \
    int ret = snprintf(msg, sizeof(msg), args...);           \
    if (ret < 128 && ret > 0) {                              \
      return Func(std::string(msg, ret));                    \
    } else {                                                 \
      return Func("Invalid message format");                 \
    }                                                        \
  }

DECLARE_ERROR(Cancelled, CANCELLED)
DECLARE_ERROR(InvalidArgument, INVALID_ARGUMENT)
DECLARE_ERROR(NotFound, NOT_FOUND)
DECLARE_ERROR(AlreadyExists, ALREADY_EXISTS)
DECLARE_ERROR(ResourceExhausted, RESOURCE_EXHAUSTED)
DECLARE_ERROR(Unavailable, UNAVAILABLE)
DECLARE_ERROR(FailedPrecondition, FAILED_PRECONDITION)
DECLARE_ERROR(OutOfRange, OUT_OF_RANGE)
DECLARE_ERROR(Unimplemented, UNIMPLEMENTED)
DECLARE_ERROR(Internal, INTERNAL)
DECLARE_ERROR(Aborted, ABORTED)
DECLARE_ERROR(DeadlineExceeded, DEADLINE_EXCEEDED)
DECLARE_ERROR(DataLoss, DATA_LOSS)
DECLARE_ERROR(Unknown, UNKNOWN)
DECLARE_ERROR(PermissionDenied, PERMISSION_DENIED)
DECLARE_ERROR(Unauthenticated, UNAUTHENTICATED)
DECLARE_ERROR(RequestStop, REQUEST_STOP)

#undef DECLARE_ERROR

::graphlearn::Status FirstErrorIfFound(
  const std::vector<::graphlearn::Status>& s);

}  // namespace error
}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_BASE_ERRORS_H_
