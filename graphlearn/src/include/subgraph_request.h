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

#ifndef GRAPHLEARN_INCLUDE_SUBGRAPH_REQUEST_H_
#define GRAPHLEARN_INCLUDE_SUBGRAPH_REQUEST_H_

#include <string>
#include "include/constants.h"
#include "include/op_request.h"

namespace graphlearn {

class SubGraphRequest : public OpRequest {
public:
  SubGraphRequest();
  SubGraphRequest(const std::string& nbr_type,
                  const std::vector<int32_t>& num_nbrs=std::vector<int32_t>(1),
                  bool need_dist=false);
  virtual ~SubGraphRequest() = default;

  OpRequest* Clone() const override;

  void Init(const Tensor::Map& params) override;
  void Set(const Tensor::Map& tensors, const SparseTensor::Map& sparse_tensors={}) override;
  void Set(const int64_t* src_id, int32_t batch_size);
  void Set(const int64_t* src_id, const int64_t* dst_id, int32_t batch_size);

  const std::string& NbrType() const;
  std::vector<int32_t> GetNumNbrs() const;
  bool NeedDist() const;
  const int64_t* GetSrcIds() const;
  const int32_t BatchSize() const;

protected:
  void Finalize() override;
  Tensor* src_ids_;
  int32_t batch_size_;
};

class SubGraphResponse : public OpResponse {
public:
  SubGraphResponse();
  virtual ~SubGraphResponse() = default;

  OpResponse* New() const override {
    return new SubGraphResponse;
  }

  void Swap(OpResponse& right) override;

  void Init(int32_t batch_size);
  void SetNodeIds(const int64_t* begin, int32_t size);
  void AppendEdge(int32_t row_idx, int32_t col_idx, int64_t e_id);
  void SetDistToSrc(const int32_t* begin, int32_t size);
  void SetDistToDst(const int32_t* begin, int32_t size);

  int32_t NodeCount() const { return batch_size_; }
  int32_t EdgeCount() const;
  const int64_t* NodeIds() const;
  const int32_t* RowIndices() const;
  const int32_t* ColIndices() const;
  const int64_t* EdgeIds() const;
  const int32_t* DistToSrc() const;
  const int32_t* DistToDst() const;

protected:
  void Finalize() override;

private:
  Tensor* node_ids_;
  Tensor* row_indices_;
  Tensor* col_indices_;
  Tensor* edge_ids_;
  Tensor* dist_to_src_;
  Tensor* dist_to_dst_;
};


}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_SUBGRAPH_REQUEST_H_
