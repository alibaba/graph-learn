/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef DGS_CORE_IO_RECORD_SLICE_H_
#define DGS_CORE_IO_RECORD_SLICE_H_

#include <string>

#include "flatbuffers/flatbuffers.h"

#include "common/slice.h"
#include "core/io/record_view.h"

namespace dgs {
namespace io {

/// A \RecordSlice corresponds to a memory buffer slice of
/// a serialized \RecordRep bytes.
///
/// During the use of a \RecordSlice, the real underlying buffer
/// referenced by it should remain valid.
class RecordSlice : public Slice {
public:
  RecordSlice() : Slice() {}
  RecordSlice(const char* d, size_t n) : Slice(d, n) {}
  RecordSlice(const uint8_t* d, size_t n)
    : Slice(reinterpret_cast<const char*>(d), n) {}
  explicit RecordSlice(const std::string& s) : Slice(s) {}

  inline RecordView GetView() const;
};

}  // namespace io
}  // namespace dgs

#endif  // DGS_CORE_IO_RECORD_SLICE_H_
