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

#ifndef GRAPHLEARN_INCLUDE_STATUS_H_
#define GRAPHLEARN_INCLUDE_STATUS_H_

#include <functional>
#include <iosfwd>
#include <memory>
#include <string>

namespace graphlearn {

namespace error {

/// For simple, we adjust the error code based on grpc StatusCode.
enum Code {
  OK                  = 0,
  CANCELLED           = 1,
  UNKNOWN             = 2,
  INVALID_ARGUMENT    = 3,
  DEADLINE_EXCEEDED   = 4,
  NOT_FOUND           = 5,
  ALREADY_EXISTS      = 6,
  PERMISSION_DENIED   = 7,
  RESOURCE_EXHAUSTED  = 8,
  FAILED_PRECONDITION = 9,
  ABORTED             = 10,
  OUT_OF_RANGE        = 11,
  UNIMPLEMENTED       = 12,
  INTERNAL            = 13,
  UNAVAILABLE         = 14,
  DATA_LOSS           = 15,
  UNAUTHENTICATED     = 16,
  REQUEST_STOP        = 17,
};

}  // namespace error

class Status {
public:
  explicit Status(error::Code code = error::OK, const char* msg = NULL);
  Status(error::Code code, const std::string& msg);
  Status(const Status& s);
  ~Status();

  Status& operator=(error::Code code);
  Status& operator=(const Status& s);
  Status& Assign(error::Code code, const char* msg = NULL);

  static Status OK() { return Status(); }

  bool ok() const { return code_ == error::OK; }
  error::Code code() const { return code_; }
  std::string msg() const;

  std::string ToString() const;

private:
  char* CopyMessage(const char* msg);

  // not allowed to compare
  bool operator==(const Status& other);

private:
  error::Code code_;
  char*       msg_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_STATUS_H_
