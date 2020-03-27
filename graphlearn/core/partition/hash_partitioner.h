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

#ifndef GRAPHLEARN_CORE_PARTITION_HASH_PARTITIONER_H_
#define GRAPHLEARN_CORE_PARTITION_HASH_PARTITIONER_H_

#include <cstdint>
#include <vector>
#include "graphlearn/core/partition/partitioner.h"
#include "graphlearn/include/config.h"
#include "graphlearn/include/op_request.h"

namespace graphlearn {

template<class T>
class HashPartitioner : public BasePartitioner<T> {
public:
  explicit HashPartitioner(int32_t range) : range_(range) {}
  ~HashPartitioner() = default;

  ShardsPtr<T> Partition(const T* req) override {
    ShardsPtr<T> ret(new Shards<T>(range_));

    if (!req->HasPartitionKey()) {
      ret->Add(GLOBAL_FLAG(ServerId), const_cast<T*>(req), false);
      return ret;
    }

    const Tensor& part_key = req->tensors_.at(req->PartitionKey());
    int32_t length = part_key.Size();
    auto part_by = part_key.GetInt64();
    for (int64_t index = 0; index < length; ++index) {
      int32_t part_id = HashToPartId(part_by[index]);

      T* part_req = ret->AddSticker(part_id, index);
      if (part_req == nullptr) {
        part_req = InitPartRequest(req);
        ret->Add(part_id, part_req, true);
      }

      for (const auto& it : req->tensors_) {
        if (it.first != kPartitionKey) {
          const Tensor* from = &(it.second);
          Tensor* target = &(part_req->tensors_[it.first]);
          CopyToPartRequest(from, index, from->Size() / length, target);
        }
      }
    }
    return ret;
  }

private:
  int32_t HashToPartId(int64_t id) {
    return llabs(id) % range_;
  }

  T* InitPartRequest(const T* req) {
    T* part_req = req->Clone();
    part_req->DisableShard();

    part_req->tensors_.reserve(req->tensors_.size());
    for (const auto& it : req->tensors_) {
      if (it.first != kPartitionKey) {
        auto& t = it.second;
        ADD_TENSOR(part_req->tensors_, it.first, t.DType() , t.Size());
      }
    }
    return part_req;
  }

  void CopyToPartRequest(const Tensor* from,
                         int32_t index,
                         int32_t dim,
                         Tensor* target) {
    DataType type = from->DType();
    int32_t end = (index + 1) * dim;

#define CASE_COPY(Type)                           \
  case k##Type:                                   \
    for (int32_t i = index * dim; i < end; ++i) { \
      target->Add##Type(from->Get##Type(i));      \
    }                                             \
    break

    switch (type) {
      CASE_COPY(Int64);
      CASE_COPY(Int32);
      CASE_COPY(Float);
      CASE_COPY(Double);
      CASE_COPY(String);
      default:
        break;
    }
#undef CASE_COPY
  }

private:
  int32_t range_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_PARTITION_HASH_PARTITIONER_H_
