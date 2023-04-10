/* Copyright 2020-2023 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef GRAPHLEARN_CORE_PARTITION_STITCHER_H_
#define GRAPHLEARN_CORE_PARTITION_STITCHER_H_

#include <cstdint>
#include <vector>
#include "include/op_request.h"

namespace graphlearn {

template<class T>
class Stitcher {
public:
  Stitcher() {}
  virtual ~Stitcher() = default;

  virtual void Stitch(ShardsPtr<T> shards, T* t) {
    int32_t shard_id = 0;
    T* tmp = nullptr;
    if (!shards->Next(&shard_id, &tmp)) {
      return;
    }

    shards->ResetNext();

    if (shards->Size() == 1) {
      t->Swap(*tmp);
    } else {
      DoStitch(shards, t);
    }
  }

private:
  void DoStitch(ShardsPtr<T> shards, T*t) {
    // Init tensors and sparse tensors, while sparse tensor
    // has been filled with segments.
    Init(shards, t);

    int32_t shard_id = 0;
    T* tmp = nullptr;

    shards->Next(&shard_id, &tmp);
    std::unordered_map<std::string, std::vector<int32_t>> to_offsets;
    for (auto& it : t->sparse_tensors_) {
      to_offsets.emplace(it.first, std::vector<int32_t>{0});
      for (int32_t i = 0; i < t->batch_size_; ++i) {
        to_offsets[it.first].push_back(
          to_offsets[it.first].back() + it.second.Segments().GetInt32(i));
      }
    }

    shards->ResetNext();
    while (shards->Next(&shard_id, &tmp)) {
      // init for sparse tensors in each shard
      // key: sparse tensor name, value: copy value from offset
      std::unordered_map<std::string, int32_t> from_offsets;
      for (auto& it : tmp->sparse_tensors_) {
        from_offsets.emplace(it.first, 0);
      }

      auto sticker = shards->StickerPtr()->At(shard_id);
      int32_t bs = tmp->batch_size_;
      if (bs == -1) {
        bs = sticker.size();
      }

      // for loop all the data in this shard
      for (int32_t i = 0; i < bs; ++i) {
        for (auto& it : tmp->tensors_) {
          // for loop all the dense tensors
          int32_t dim = it.second.Size() / bs;
          int32_t from_offset = i * dim;
          int32_t to_offset = sticker[i] * dim;
          CopyToResponse(&(it.second),
                          from_offset,
                          &(t->tensors_[it.first]),
                          to_offset,
                          dim);
        }
        for (auto& it : tmp->sparse_tensors_) {
          // for loop all the sparse tensors
          int32_t offset = sticker[i];
          auto& segments = it.second.Segments();
          auto& values = it.second.Values();
          int32_t dim = segments.GetInt32(i);
          auto from_offset = from_offsets[it.first];
          auto& to = t->sparse_tensors_[it.first];
          auto to_offset = to_offsets[it.first][offset];
          CopyToResponse(&values, from_offset, to.MutableValues(), to_offset, dim);
          from_offsets[it.first] = from_offset + dim;
        }
      }
    }
  }

  void Init(ShardsPtr<T> shards, T* t) {
    int32_t shard_id = 0;
    T* tmp = nullptr;
    shards->Next(&shard_id, &tmp);

    int32_t batch_size = shards->StickerPtr()->Size();
    auto sticker = shards->StickerPtr()->At(shard_id);
    int32_t bs = tmp->batch_size_;
    if (bs == -1 ) {
      bs = sticker.size();
    }
    t->batch_size_ = batch_size;
    t->params_ = tmp->params_;
    t->tensors_.reserve(tmp->tensors_.size());
    for (auto& it : tmp->tensors_) {
      int32_t dim = it.second.Size() / bs;
      ADD_TENSOR(t->tensors_, it.first, it.second.DType(), batch_size * dim);
      t->tensors_[it.first].Resize(batch_size * dim);
    }
    shards->ResetNext();

    auto sparse_tensors_num = tmp->sparse_tensors_.size();
    if (sparse_tensors_num == 0) {
      return;
    }
    t->sparse_tensors_.reserve(sparse_tensors_num);

    std::unordered_map<std::string, int32_t> value_size_map;
    std::unordered_map<std::string, Tensor> to_segments_map;
    std::unordered_map<std::string, Tensor> to_values_map;

    for (auto& from : tmp->sparse_tensors_) {
      auto& from_segments = from.second.Segments();
      auto& from_values = from.second.Values();
      value_size_map.emplace(from.first, 0);
      Tensor to_segments(from_segments.DType(), batch_size);
      to_segments.Resize(batch_size);
      to_segments_map.emplace(from.first, std::move(to_segments));
      Tensor to_values(from_values.DType());
      to_values_map.emplace(from.first, std::move(to_values));
      // to_values size is not set now.
    }

    while (shards->Next(&shard_id, &tmp)) {
      auto stickers = shards->StickerPtr()->At(shard_id);
      for (auto& from : tmp->sparse_tensors_) {
        for (int32_t i = 0; i < tmp->batch_size_; ++i) {
          int32_t offset = stickers[i];
          auto& from_segments = from.second.Segments();
          to_segments_map[from.first].SetInt32(offset, from_segments.GetInt32(i));
          value_size_map[from.first] += from_segments.GetInt32(i);
        }
      }
    }

    for (auto& iter : to_values_map) {
      iter.second.Resize(value_size_map[iter.first]);
      t->sparse_tensors_.emplace(iter.first,
        std::move(SparseTensor{
          std::move(to_segments_map[iter.first]), std::move(to_values_map[iter.first])}));
    }

    shards->ResetNext();
  }

  void CopyToResponse(const Tensor* from, int32_t from_offset,
                      Tensor* to, int32_t to_offset,
                      int32_t dim) {
#define CASE_COPY(type, from, from_offset, to, to_offset, dim)        \
  case k##type:                                                       \
    for (int32_t i = 0; i < dim; ++i) {                               \
      to->Set##type(to_offset + i, from->Get##type(from_offset + i)); \
    }                                                                 \
    break

    DataType type = from->DType();
    switch (type) {
      CASE_COPY(Int64, from, from_offset, to, to_offset, dim);
      CASE_COPY(Int32, from, from_offset, to, to_offset, dim);
      CASE_COPY(Float, from, from_offset, to, to_offset, dim);
      CASE_COPY(Double, from, from_offset, to, to_offset, dim);
      CASE_COPY(String, from, from_offset, to, to_offset, dim);
      default:
        break;
    }
#undef CASE_COPY
  }
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_PARTITION_STITCHER_H_
