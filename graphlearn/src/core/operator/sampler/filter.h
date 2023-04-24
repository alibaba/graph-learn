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

#ifndef GRAPHLEARN_CORE_OPERATOR_SAMPLER_FILTER_H_
#define GRAPHLEARN_CORE_OPERATOR_SAMPLER_FILTER_H_

#include <functional>
#include "include/constants.h"
#include "include/tensor.h"
#include "core/graph/storage/graph_storage.h"
#include "core/graph/storage/types.h"

namespace graphlearn {
namespace op {

/// Filter is for NeighborSampler, when filter is set,
/// the sampled neighbors will not conclude the nodes and edges whose
/// field(such as Id, Timestamp, etc.) is LargerThan(or Equal to, etc.
/// depends on FilterType) the filter value.
class Filter {
  using FieldFunc = std::function<int64_t(
    io::GraphStorage* storage,
    io::IdType nbr_id, io::IdType edge_ids)>;

  using FilterFunc = std::function<bool(
    io::GraphStorage* storage, int batch_idx,
    const io::Array<io::IdType>& nbr_ids,
    const io::Array<io::IdType>& edge_ids,
    int nbr_idx, Tensor* values)>;

public:
  Filter();
  Filter(FilterField field, FilterType type);

  Filter(Filter&& filter) noexcept;
  Filter(const Filter& filter) = delete;
  Filter& operator=(Filter&& filter) noexcept;
  Filter& operator=(const Filter& filter) = delete;

  ~Filter() = default;

  void InitValues(Tensor* values);

  /// Fill filter values for the bacth of src_ids in SamplingRequest and
  /// align the values with src_ids.
  /// For example, when sampling neighbors C for batch nodes B, and filtering out
  /// the ids in batch nodes A. Since each node in A corresponds to 2
  /// neighbor nodes in B, so it is required to expand A(filter_values)
  /// and align it with the size of B(src_ids).
  /// src:     A
  ///         / |
  /// 1hop:  B   B
  ///        /| /|
  /// 2hop: C C C C
  void FillValues(const Tensor& values, int32_t size);

  FilterType GetType() const { return type_; }
  FilterField GetField() const { return field_; }
  Tensor* GetValue() const { return values_; }

  operator bool () const {
    return type_ != FilterType::kOperatorUnspecified;
  }

  /// Act filter on neighbors of src_ids[batch_idx], and return the reserved
  /// neighbors' indices.
  void ActOn(int32_t batch_idx,
             const io::Array<io::IdType>& nbr_ids,
             const io::Array<io::IdType>& edge_ids,
             io::GraphStorage* storage,
             std::vector<io::IndexType>* indices) const;

  /// Check if the field value for neighbors[nbr_idx] of src_ids[batch_idx]
  /// hit the corresponding filter value.
  bool Hit(int32_t batch_idx,
           const io::Array<io::IdType>& nbr_ids,
           const io::Array<io::IdType>& edge_ids,
           int nbr_idx,
           io::GraphStorage* storage) const;

  /// Check if the field values for all neighbors of src_ids[batch_idx]
  /// hit the corresponding filter value.
  bool HitAll(int32_t batch_idx,
              const io::Array<io::IdType>& nbr_ids,
              const io::Array<io::IdType>& edge_ids,
              io::GraphStorage* storage) const;

private:
  FieldFunc GetFieldFunc() const;
  FilterFunc GetFilterFunc() const;

  /// When the filter field values are ascending, find the largest position
  /// of neighbors whose field is smaller than the filter value.
  /// If batch_share_idx < 0, each src_id in the batch uses filter value
  /// of corrsponding batch_idx. Else, the batch share the idx(th) filter value.
  int FindkthLargest(int32_t batch_idx,
                     const io::Array<io::IdType>& nbr_ids,
                     const io::Array<io::IdType>& edge_ids,
                     io::GraphStorage* storage,
                     int32_t batch_share_idx=0) const;

private:
  FilterType   type_;
  FilterField  field_;
  Tensor*      values_;  // Batch of filter values, corresponding to src_ids.
  FilterFunc   filter_func_;
};

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_OPERATOR_SAMPLER_FILTER_H_
