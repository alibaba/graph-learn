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

#include "graphlearn/common/base/errors.h"

namespace graphlearn {
namespace error {

#define DEFINE_ERROR(Func, Type)                                 \
  ::graphlearn::Status Func(const std::string& msg) {            \
    return ::graphlearn::Status(::graphlearn::error::Type, msg); \
  }

DEFINE_ERROR(Cancelled, CANCELLED)
DEFINE_ERROR(InvalidArgument, INVALID_ARGUMENT)
DEFINE_ERROR(NotFound, NOT_FOUND)
DEFINE_ERROR(AlreadyExists, ALREADY_EXISTS)
DEFINE_ERROR(ResourceExhausted, RESOURCE_EXHAUSTED)
DEFINE_ERROR(Unavailable, UNAVAILABLE)
DEFINE_ERROR(FailedPrecondition, FAILED_PRECONDITION)
DEFINE_ERROR(OutOfRange, OUT_OF_RANGE)
DEFINE_ERROR(Unimplemented, UNIMPLEMENTED)
DEFINE_ERROR(Internal, INTERNAL)
DEFINE_ERROR(Aborted, ABORTED)
DEFINE_ERROR(DeadlineExceeded, DEADLINE_EXCEEDED)
DEFINE_ERROR(DataLoss, DATA_LOSS)
DEFINE_ERROR(Unknown, UNKNOWN)
DEFINE_ERROR(PermissionDenied, PERMISSION_DENIED)
DEFINE_ERROR(Unauthenticated, UNAUTHENTICATED)
DEFINE_ERROR(RequestStop, REQUEST_STOP)

#undef DEFINE_ERROR

::graphlearn::Status FirstErrorIfFound(
    const std::vector<::graphlearn::Status>& s) {
  for (size_t i = 0; i < s.size(); ++i) {
    if (!s[i].ok()) {
      return s[i];
    }
  }
  return ::graphlearn::Status::OK();
}

}  // namespace error
}  // namespace graphlearn
