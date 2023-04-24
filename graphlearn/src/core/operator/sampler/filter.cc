/* Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "core/operator/sampler/filter.h"

namespace graphlearn {
namespace op {

Filter::Filter() : type_(FilterType::kOperatorUnspecified),
                   field_(FilterField::kFieldUnspecified),
                   values_(nullptr) {
  filter_func_ = std::move(GetFilterFunc());
}

Filter::Filter(FilterField field, FilterType type)
  : type_(type), field_(field), values_(nullptr) {
  filter_func_ = std::move(GetFilterFunc());
}

Filter::Filter(Filter&& rhs) noexcept
  : type_(rhs.type_), field_(rhs.field_), values_(rhs.values_),
    filter_func_(std::move(rhs.filter_func_)) {
  rhs.values_ = nullptr;
}

Filter& Filter::operator=(Filter&& rhs) noexcept {
  if (this != &rhs) {
    type_ = rhs.type_;
    field_ = rhs.field_;
    values_ = rhs.values_;
    rhs.values_ = nullptr;
    filter_func_ = std::move(rhs.filter_func_);
  }
  return *this;
}

void Filter::InitValues(Tensor* values) {
  values_ = values;
}

void Filter::FillValues(const Tensor& values, int32_t size) {
  if (type_ != FilterType::kOperatorUnspecified) {
    const int64_t* filter_values = values.GetInt64();
    int32_t filter_size = values.Size();
    if (filter_size != 0) {
      int32_t fanout = size / filter_size;
      for (int32_t i = 0; i < filter_size; ++i) {
        for (int32_t j = 0; j < fanout; ++j) {
          values_->AddInt64(*(filter_values + i));
        }
      }
    }
  }
}

void Filter::ActOn(int32_t batch_idx,
                   const io::Array<io::IdType>& nbr_ids,
                   const io::Array<io::IdType>& edge_ids,
                   io::GraphStorage* storage,
                   std::vector<io::IndexType>* indices) const {
  if (field_ == FilterField::kTimestamp && type_ == FilterType::kLargerThan) {
    // Accelerating filter on ascending filed(timestamps).
    int k = FindkthLargest(batch_idx, nbr_ids, edge_ids, storage);
    if (k < 0) {
      k = 0;
    }
    indices->resize(k);
    std::reverse(indices->begin(), indices->end());
    // return with descending order.
  } else {
    int l = 0;
    int r = nbr_ids.Size() - 1;
    while (l <= r) {
      while (Hit(batch_idx, nbr_ids, edge_ids, indices->at(l), storage) && (l <= r)) {
        std::iter_swap(indices->begin() + l, indices->begin() + r);
        --r;
      }
      ++l;
    }
    indices->resize(r + 1);
  }
}

bool Filter::Hit(int32_t batch_idx,
                 const io::Array<io::IdType>& nbr_ids,
                 const io::Array<io::IdType>& edge_ids,
                 int nbr_idx,
                 io::GraphStorage* storage) const {
  if (type_ == FilterType::kOperatorUnspecified) {
    return false;
  }
  return filter_func_(storage, batch_idx,
                      nbr_ids, edge_ids, nbr_idx, values_);
}

bool Filter::HitAll(int32_t batch_idx,
                    const io::Array<io::IdType>& nbr_ids,
                    const io::Array<io::IdType>& edge_ids,
                    io::GraphStorage* storage) const {
  if (type_ == FilterType::kOperatorUnspecified) {
    return false;
  }
  bool res = true;
  for (int i = 0; i < nbr_ids.Size(); ++i) {
    res &= filter_func_(storage, batch_idx,
                        nbr_ids, edge_ids, i, values_);
    if (!res) {
      return false;
    }
  }
  return true;
}

Filter::FieldFunc Filter::GetFieldFunc() const {
  FieldFunc get_field = [] (
      io::GraphStorage* storage,
      io::IdType nbr_id, io::IdType edge_id) {
    return -1;
  };
  switch (field_)
  {
    case FilterField::kId: {
      get_field = [] (io::GraphStorage* storage,
                      io::IdType nbr_id, io::IdType edge_id) {
        return nbr_id;
      };
    }
    break;
    case FilterField::kTimestamp: {
      get_field = [] (io::GraphStorage* storage,
                      io::IdType nbr_id, io::IdType edge_id) {
        return storage->GetEdgeTimestamp(edge_id);
      };
    }
    break;
    default:
    break;
  }
  return get_field;
}

Filter::FilterFunc Filter::GetFilterFunc() const {
  Filter::FilterFunc filter_func = [] (
      io::GraphStorage* storage,
      int batch_idx,
      const io::Array<io::IdType>& nbr_ids, const io::Array<io::IdType>& edge_ids,
      int nbr_idx, Tensor* values) {
    return false;
  };

  FieldFunc get_field = GetFieldFunc();

  switch (type_)
  {
    case FilterType::kEqual: {
      filter_func = [get_field] (
          io::GraphStorage* storage,
          int batch_idx,
          const io::Array<io::IdType>& nbr_ids, const io::Array<io::IdType>& edge_ids,
          int nbr_idx, Tensor* values) {
        auto value = get_field(storage, nbr_ids[nbr_idx], edge_ids[nbr_idx]);
        return value == values->GetInt64(batch_idx);
      };
    }
    break;
    case FilterType::kLargerThan: {
      filter_func = [get_field] (
          io::GraphStorage* storage,
          int batch_idx,
          const io::Array<io::IdType>& nbr_ids, const io::Array<io::IdType>& edge_ids,
          int nbr_idx, Tensor* values) {
        auto value = get_field(storage, nbr_ids[nbr_idx], edge_ids[nbr_idx]);
        return value > values->GetInt64(batch_idx);
      };
    }
    break;
    default:
    break;
  }
  return filter_func;
}

int Filter::FindkthLargest(int32_t batch_idx,
                           const io::Array<io::IdType>& nbr_ids,
                           const io::Array<io::IdType>& edge_ids,
                           io::GraphStorage* storage,
                           int32_t batch_share_idx) const {
  int idx = batch_idx;
  if (batch_share_idx >= 0) {
    idx = batch_share_idx;
  }
  int start = 0;
  int end = nbr_ids.Size() - 1;
  auto filter = values_->GetInt64(idx);
  if (end == 0) {
    return -1;
  }
  int mid = 0;
  FieldFunc get_field = GetFieldFunc();

  while (end >= start) {
    mid = start + (end - start) / 2;
    auto value = get_field(storage, nbr_ids[mid], edge_ids[mid]);
    if (value == filter) {
      return mid;
    }
    else if (value > filter ) {
      end = mid - 1;
    } else {
      start = mid + 1;
    }
  }
  if (get_field(storage, nbr_ids[mid], edge_ids[mid]) < filter) {
    mid += 1;
  }
  return mid;
}

}  // namespace op
}  // namespace graphlearn

