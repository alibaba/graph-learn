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

#ifndef GRAPHLEARN_ACTOR_GRAPH_WRAPPER_PROTO_H_
#define GRAPHLEARN_ACTOR_GRAPH_WRAPPER_PROTO_H_

namespace graphlearn {
namespace act {

#include <iostream>
#include <streambuf>
#include <string>
#include <utility>
#include "brane/actor/actor_message.hh"
#include "seastar/core/deleter.hh"
#include "seastar/core/temporary_buffer.hh"

template <typename ProtoData>
class PbWrapper {
  struct membuf : std::streambuf {
    membuf(char* begin, char* end) {
      this->setg(begin, begin, end);
    }
  };

public:
  PbWrapper() : data() {}

  explicit PbWrapper(ProtoData& other) : data() {
    data.Swap(&other);
  }

  PbWrapper(PbWrapper&& other) noexcept : data() {
    data.Swap(&other.data);
  }

  PbWrapper(const PbWrapper &other) = delete;

  void dump_to(brane::serializable_queue &qu) {  // NOLINT [runtime/references]
    std::string bytes;
    data.SerializeToString(&bytes);
    char* bytes_ptr = const_cast<char*>(bytes.data());
    auto length = bytes.size();
    qu.push(seastar::temporary_buffer<char>(bytes_ptr, length,
      seastar::make_object_deleter(std::move(bytes))));
  }

  static PbWrapper
  load_from(brane::serializable_queue& qu) {  // NOLINT [runtime/references]
    auto buf = qu.pop();
    char *ptr = buf.get_write();
    membuf sbuf(ptr, ptr + buf.size());
    std::istream is(&sbuf);
    PbWrapper<ProtoData> pw;
    pw.data.ParseFromIstream(&is);
    return pw;
  }

  ProtoData data;
};

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_GRAPH_WRAPPER_PROTO_H_
