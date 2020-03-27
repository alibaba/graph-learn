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

#ifndef GRAPHLEARN_COMMON_IO_LINE_ITERATOR_H_
#define GRAPHLEARN_COMMON_IO_LINE_ITERATOR_H_

#include <string>
#include "graphlearn/common/base/macros.h"
#include "graphlearn/common/base/uncopyable.h"
#include "graphlearn/include/status.h"
#include "graphlearn/platform/env.h"

namespace graphlearn {
namespace io {

class LineIterator : private Uncopyable {
public:
  LineIterator(ByteStreamAccessFile* file, size_t buffer_bytes);
  virtual ~LineIterator();

  // Read a line of data into "*result" until "\n" is got.
  // Overwrites any existing data in *result.
  // If successful, return OK.
  // If reach the end of the file, return OUT_OF_RANGE.
  Status Next(std::string* result);

protected:
  virtual Status FillBuffer();

  ByteStreamAccessFile* file_;
  size_t size_;
  char*  buf_;
  char*  pos_;
  char*  limit_;
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_IO_LINE_ITERATOR_H_
