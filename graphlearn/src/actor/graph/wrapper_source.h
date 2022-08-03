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

#ifndef GRAPHLEARN_ACTOR_GRAPH_WRAPPER_SOURCE_H_
#define GRAPHLEARN_ACTOR_GRAPH_WRAPPER_SOURCE_H_

#include <string>
#include <utility>
#include <vector>
#include "brane/net/serializable_queue.hh"
#include "common/io/value.h"
#include "core/io/parser.h"
#include "include/data_source.h"

namespace graphlearn {
namespace actor {

namespace {  // NOLINT [build/namespaces]

template <typename U>
void Parse(const U* source, io::SideInfo* info) {}

template <>
void Parse(const io::EdgeSource* source, io::SideInfo* info) {
  ParseSideInfo(source, info);
  info->type = source->edge_type;
}

template <>
void Parse(const io::NodeSource* source, io::SideInfo* info) {
  ParseSideInfo(source, info);
  info->type = source->id_type;
}

}  // anonymous namespace

template <class T>
class SourceWrapper {
public:
  SourceWrapper() : impl_(nullptr) {}

  explicit SourceWrapper(T* impl) : impl_(impl) {
    if (Valid()) {
      Parse(impl_, &info_);
    }
  }

  SourceWrapper(SourceWrapper&& rhs) {
    impl_ = rhs.impl_;
    rhs.impl_ = nullptr;
    info_.CopyFrom(rhs.info_);
  }

  SourceWrapper& operator=(SourceWrapper&& rhs) {
    if (this != &rhs) {
      impl_ = rhs.impl_;
      rhs.impl_ = nullptr;
      info_.CopyFrom(rhs.info_);
    }
    return *this;
  }

  bool Valid() const {
    return impl_ != nullptr;
  }

  bool IsWeighted() const {
    return impl_->IsWeighted();
  }

  bool IsLabeled() const {
    return impl_->IsLabeled();
  }

  bool IsAttributed() const {
    return impl_->IsAttributed();
  }

  const io::AttributeInfo& GetAttributeInfo() const {
    return impl_->attr_info;
  }

  const io::SideInfo& GetSideInfo() const {
    return info_;
  }

  // Just need the interface to compile pass
  void dump_to(brane::serializable_queue &su) {  // NOLINT [runtime/references]
  }

  // Just need the interface to compile pass
  static SourceWrapper<T>
  load_from(brane::serializable_queue& su) {  // NOLINT [runtime/references]
    return SourceWrapper<T>();
  }

private:
  T* impl_;
  io::SideInfo info_;
};

using NodeSourceWrapper = SourceWrapper<io::NodeSource>;
using EdgeSourceWrapper = SourceWrapper<io::EdgeSource>;

}  // namespace actor
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_GRAPH_WRAPPER_SOURCE_H_
