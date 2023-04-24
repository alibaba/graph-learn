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

#ifndef GRAPHLEARN_INCLUDE_SAMPLING_REQUEST_H_
#define GRAPHLEARN_INCLUDE_SAMPLING_REQUEST_H_

#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>
#include "include/constants.h"
#include "include/op_request.h"
#include "core/operator/sampler/filter.h"

namespace graphlearn {

///  This struct is for unified expression of dense shape and spase shape
///  for neighbor sampling.
struct Shape {
  size_t dim1; // batch size
  size_t dim2; // max neighbor count
  size_t size; // total neighbor count
  std::vector<int32_t> segments; // true neighbor count for each in batch
  bool sparse; // is sparse or not

  Shape()
    : dim1(0), dim2(0), size(0), segments(), sparse(false) {}
  Shape(size_t x, size_t y) : dim1(x), dim2(y), size(x * y), segments(x, y), sparse(false) {}
  Shape(size_t x, size_t y, const std::vector<int32_t> inds)
      : dim1(x), dim2(y),
        size(std::accumulate(inds.begin(), inds.end(), 0)),
        segments(inds),
        sparse(true) {
  }

  Shape(const Shape& other)
    : dim1(other.dim1), dim2(other.dim2),
      size(other.size), segments(other.segments), sparse(other.sparse) {}

  Shape(Shape&& other)
    : dim1(other.dim1), dim2(other.dim2),
      size(other.size), segments(std::move(other.segments)), sparse(other.sparse) {}

  Shape& operator=(const Shape& other) {
    dim1 = other.dim1;
    dim2 = other.dim2;
    size = other.size;
    segments = other.segments;
    sparse = other.sparse;
    return *this;
  }

  Shape& operator=(Shape&& other) {
    dim1 = other.dim1;
    dim2 = other.dim2;
    size = other.size;
    segments = std::move(other.segments);
    sparse = other.sparse;
    return *this;
  }

  void Swap(Shape& other) {
    std::swap(dim1, other.dim1);
    std::swap(dim2, other.dim2);
    std::swap(size, other.size);
    segments.swap(other.segments);
    std::swap(sparse, other.sparse);
  }
};

class SamplingRequest : public OpRequest {
public:
  SamplingRequest();
  SamplingRequest(const std::string& type,
                  const std::string& strategy,
                  int32_t neighbor_count,
                  FilterType filter_type = FilterType::kOperatorUnspecified,
                  FilterField filter_field = FilterField::kFieldUnspecified);
  ~SamplingRequest() = default;

  OpRequest* Clone() const override;

  void Init(const Tensor::Map& params) override;
  void Set(const Tensor::Map& tensors, const SparseTensor::Map& sparse_tensors={}) override;
  void Set(const int64_t* src_ids, int32_t batch_size);

  const std::string& Type() const;
  const std::string& Strategy() const;
  int32_t BatchSize() const;
  int32_t NeighborCount() const { return neighbor_count_; }
  const int64_t* GetSrcIds() const;

  const op::Filter* GetFilter() const;

protected:
  void Finalize() override;
  int32_t neighbor_count_;
  Tensor* src_ids_;
  op::Filter  filter_;
};

class SamplingResponse : public OpResponse {
public:
  SamplingResponse();
  ~SamplingResponse() = default;

  OpResponse* New() const override {
    return new SamplingResponse;
  }

  void Swap(OpResponse& right) override;

  void SetShape(size_t dim1, size_t dim2);
  void SetShape(size_t dim1, size_t dim2, const std::vector<int32_t>& segments);
  void SetShape(size_t dim1, size_t dim2, std::vector<int32_t>&& segments);
  void InitNeighborIds();
  void InitEdgeIds();

  void AppendNeighborId(int64_t id);
  void AppendEdgeId(int64_t id);
  void AppendDegree(int32_t degree);
  void FillWith(int64_t neighbor_id, int64_t edge_id = -1);

  const Shape GetShape() const;
  int64_t* GetNeighborIds();
  int64_t* GetEdgeIds();
  const int64_t* GetNeighborIds() const;
  const int64_t* GetEdgeIds() const;

protected:
  void Finalize() override;

private:
  Shape shape_; // Get degrees for SparseTensor from shape
  Tensor* neighbors_;
  Tensor* edges_;
};


class ConditionalSamplingRequest : public SamplingRequest {
public:
  ConditionalSamplingRequest();
  ConditionalSamplingRequest(const std::string& type,
                             const std::string& strategy,
                             int32_t neighbor_count,
                             const std::string& dst_node_type,
                             bool batch_share,
                             bool unique);
  ~ConditionalSamplingRequest() = default;

  OpRequest* Clone() const override;

  // For DagNodeRunner.
  void Init(const Tensor::Map& params) override;
  void Set(const Tensor::Map& tensors, const SparseTensor::Map& sparse_tensors={}) override;

  void SetIds(const int64_t* src_ids,
              const int64_t* dst_ids,
              int32_t batch_size);
  void SetSelectedCols(const std::vector<int32_t>& int_cols,
                       const std::vector<float>& int_props,
                       const std::vector<int32_t>& float_cols,
                       const std::vector<float>& float_props,
                       const std::vector<int32_t>& str_cols,
                       const std::vector<float>& str_props);
  const std::string& Strategy() const;
  const std::string& DstNodeType() const;
  const bool BatchShare() const;
  const bool Unique() const;
  const int64_t* GetDstIds() const;

  const std::vector<int32_t> IntCols() const;
  const std::vector<float> IntProps() const;
  const std::vector<int32_t> FloatCols() const;
  const std::vector<float> FloatProps() const;
  const std::vector<int32_t> StrCols() const;
  const std::vector<float> StrProps() const;

protected:
  void Finalize() override;

private:
  Tensor* dst_ids_;
  Tensor* int_cols_;
  Tensor* int_props_;
  Tensor* float_cols_;
  Tensor* float_props_;
  Tensor* str_cols_;
  Tensor* str_props_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_SAMPLING_REQUEST_H_
