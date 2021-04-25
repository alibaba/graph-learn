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

#ifndef GRAPHLEARN_INCLUDE_SUBGRAPH_REQUEST_H_
#define GRAPHLEARN_INCLUDE_SUBGRAPH_REQUEST_H_

#include <string>
#include "graphlearn/include/constants.h"
#include "graphlearn/include/op_request.h"

namespace graphlearn {

class SubGraphRequest : public OpRequest {
public:
  SubGraphRequest();
  SubGraphRequest(const std::string& seed_type,
                  const std::string& nbr_type,
                  const std::string& strategy,
                  int32_t batch_size,
                  int32_t epoch = 0);
  virtual ~SubGraphRequest() = default;

  const std::string& SeedType() const;
  const std::string& NbrType() const;
  const std::string& Strategy() const;
  int32_t BatchSize() const;
  int32_t Epoch() const;
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
  int32_t NodeCount() const { return batch_size_; }
  int32_t EdgeCount() const;
  const int64_t* NodeIds() const;
  const int32_t* RowIndices() const;
  const int32_t* ColIndices() const;
  const int64_t* EdgeIds() const;

protected:
  void SetMembers() override;

private:
  Tensor* node_ids_;
  Tensor* row_indices_;
  Tensor* col_indices_;
  Tensor* edge_ids_;
};


}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_SUBGRAPH_REQUEST_H_
