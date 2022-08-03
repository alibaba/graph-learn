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

#ifndef GRAPHLEARN_ACTOR_GRAPH_WRAPPER_REQUEST_H_
#define GRAPHLEARN_ACTOR_GRAPH_WRAPPER_REQUEST_H_

#include <memory>
#include <string>
#include <utility>

#include "brane/actor/actor_message.hh"
#include "core/io/element_value.h"
#include "include/config.h"
#include "include/graph_request.h"
#include "proto/service.pb.h"

namespace graphlearn {
namespace actor {

template <class T, class V>
class RequestWrapper {
  struct MemBuf : std::streambuf {
    MemBuf(char* begin, char* end) {
      this->setg(begin, begin, end);
    }
  };

public:
  RequestWrapper() : impl_(nullptr), side_info_(nullptr) {
  }

  RequestWrapper(RequestWrapper&& rhs) {
    impl_ = rhs.impl_;
    side_info_ = rhs.side_info_;
    rhs.impl_ = nullptr;
    rhs.side_info_ = nullptr;
  }

  RequestWrapper& operator=(RequestWrapper&& rhs) {
    impl_ = rhs.impl_;
    side_info_ = rhs.side_info_;
    rhs.impl_ = nullptr;
    rhs.side_info_ = nullptr;
    return *this;
  }

  ~RequestWrapper() {
    delete impl_;
    delete side_info_;
  }

  void Set(const io::SideInfo& side_info) {
    // FIXME: move to constructor.
    if (side_info_) {
      delete side_info_;
    }
    if (impl_) {
      delete impl_;
    }
    side_info_ = new io::SideInfo();
    side_info_->CopyFrom(side_info);
    impl_ = new T(side_info_, GLOBAL_FLAG(DataInitBatchSize));
  }

  T* Get() {
    return impl_;
  }

  int32_t Size() {
    return impl_->Size();
  }

  void Append(const V* value) {
    impl_->Append(value);
  }

  const std::string& Type() {
    return impl_->GetSideInfo()->type;
  }

  bool Empty() const {
    return impl_ == nullptr;
  }

  void print() {
    OpRequestPb pb;
    impl_->SerializeTo(&pb);
    std::cout << "[ Request Wrapper print Protobuf ]\n"
              << pb.DebugString() << std::endl;
  }

  void dump_to(brane::serializable_queue& qu) {  // NOLINT [runtime/references]
    OpRequestPb pb;
    impl_->SerializeTo(&pb);

    std::string bytes;
    pb.SerializeToString(&bytes);
    char* bytes_ptr = const_cast<char*>(bytes.data());
    auto length = bytes.size();
    qu.push(seastar::temporary_buffer<char>(bytes_ptr, length,
      seastar::make_object_deleter(std::move(bytes))));
  }

  static RequestWrapper<T, V>
  load_from(brane::serializable_queue& qu) {  // NOLINT [runtime/references]
    auto buf = qu.pop();
    char *ptr = buf.get_write();
    MemBuf sbuf(ptr, ptr + buf.size());
    std::istream is(&sbuf);
    OpRequestPb pb;
    pb.ParseFromIstream(&is);
    RequestWrapper<T, V> rw;
    rw.impl_ = new T;
    rw.impl_->ParseFrom(&pb);
    return rw;
  }

private:
  T* impl_;
  io::SideInfo* side_info_;
};

#define REQUEST(Type) RequestWrapper<Update##Type##sRequest, io::Type##Value>

using UpdateNodesRequestWrapper = REQUEST(Node);
using UpdateEdgesRequestWrapper = REQUEST(Edge);

typedef std::shared_ptr<REQUEST(Node)> UpdateNodesPtr;
typedef std::shared_ptr<REQUEST(Edge)> UpdateEdgesPtr;

#undef REQUEST

}  // namespace actor
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_GRAPH_WRAPPER_REQUEST_H_
