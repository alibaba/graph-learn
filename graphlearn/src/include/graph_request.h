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

#ifndef GRAPHLEARN_INCLUDE_GRAPH_REQUEST_H_
#define GRAPHLEARN_INCLUDE_GRAPH_REQUEST_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "include/constants.h"
#include "include/graph_statistics.h"
#include "include/op_request.h"

namespace graphlearn {

namespace io {
struct SideInfo;
struct EdgeValue;
struct NodeValue;
struct AttributeValue;
}  // namespace io

class UpdateRequest : public OpRequest {
public:
  UpdateRequest();
  UpdateRequest(const io::SideInfo* info, int32_t batch_size);
  virtual ~UpdateRequest();

  void Append(const io::AttributeValue* value);
  const io::SideInfo* GetSideInfo() const;
  int32_t Size() const;
  void Next(io::AttributeValue* value);

protected:
  void SetMembers() override;

protected:
  io::SideInfo* info_;
  int32_t cursor_;

  Tensor* infos_;
  Tensor* weights_;
  Tensor* labels_;
  Tensor* i_attrs_;
  Tensor* f_attrs_;
  Tensor* s_attrs_;
};

class UpdateEdgesRequest : public UpdateRequest {
public:
  UpdateEdgesRequest();
  UpdateEdgesRequest(const io::SideInfo* info, int32_t batch_size);
  virtual ~UpdateEdgesRequest() = default;

  OpRequest* Clone() const override;
  void SerializeTo(void* request) override;

  int32_t Size() const;
  void Append(const io::EdgeValue* value);
  bool Next(io::EdgeValue* value);

protected:
  void SetMembers() override;

private:
  Tensor* src_ids_;
  Tensor* dst_ids_;
};

class UpdateEdgesResponse : public OpResponse {
public:
  virtual ~UpdateEdgesResponse() = default;

  OpResponse* New() const override {
    return new UpdateEdgesResponse;
  }

  void Stitch(ShardsPtr<OpResponse> shards) override {}
};

class UpdateNodesRequest : public UpdateRequest {
public:
  UpdateNodesRequest();
  UpdateNodesRequest(const io::SideInfo* info, int32_t batch_size);
  virtual ~UpdateNodesRequest() = default;

  OpRequest* Clone() const override;
  void SerializeTo(void* request) override;

  int32_t Size() const;
  void Append(const io::NodeValue* value);
  bool Next(io::NodeValue* value);

protected:
  void SetMembers() override;

private:
  Tensor* ids_;
};

class UpdateNodesResponse : public OpResponse {
public:
  virtual ~UpdateNodesResponse() = default;

  OpResponse* New() const override {
    return new UpdateNodesResponse;
  }

  void Stitch(ShardsPtr<OpResponse> shards) override {}
};

class GetEdgesRequest : public OpRequest {
public:
  GetEdgesRequest();
  GetEdgesRequest(const std::string& edge_type,
                  const std::string& strategy,
                  int32_t batch_size,
                  int32_t epoch = 0);
  virtual ~GetEdgesRequest() = default;

  void Init(const std::unordered_map<std::string, Tensor>& params) override;

  const std::string& EdgeType() const;
  const std::string& Strategy() const;
  int32_t BatchSize() const;
  int32_t Epoch() const;
};

class GetEdgesResponse : public OpResponse {
public:
  GetEdgesResponse();
  virtual ~GetEdgesResponse() = default;

  OpResponse* New() const override {
    return new GetEdgesResponse;
  }

  void Swap(OpResponse& right) override;

  void Init(int32_t batch_size);
  void Append(int64_t src_id, int64_t dst_id, int64_t edge_id);
  int32_t Size() const { return batch_size_; }
  const int64_t* SrcIds() const;
  const int64_t* DstIds() const;
  const int64_t* EdgeIds() const;

protected:
  void SetMembers() override;

private:
  Tensor* src_ids_;
  Tensor* dst_ids_;
  Tensor* edge_ids_;
};

class GetNodesRequest : public OpRequest {
public:
  GetNodesRequest();
  GetNodesRequest(const std::string& type,
                  const std::string& strategy,
                  NodeFrom node_from,
                  int32_t batch_size,
                  int32_t epoch = 0);
  virtual ~GetNodesRequest() = default;

  void Init(const std::unordered_map<std::string, Tensor>& params) override;

  const std::string& Type() const;
  const std::string& Strategy() const;
  NodeFrom GetNodeFrom() const;
  int32_t BatchSize() const;
  int32_t Epoch() const;
};

class GetNodesResponse : public OpResponse {
public:
  GetNodesResponse();
  virtual ~GetNodesResponse() = default;

  OpResponse* New() const override {
    return new GetNodesResponse;
  }

  void Swap(OpResponse& right) override;

  void Init(int32_t batch_size);
  void Append(int64_t node_id);
  int32_t Size() const { return batch_size_; }
  const int64_t* NodeIds() const;

protected:
  void SetMembers() override;

private:
  Tensor* node_ids_;
};

class LookupEdgesRequest : public OpRequest {
public:
  LookupEdgesRequest();
  explicit LookupEdgesRequest(const std::string& edge_type,
                              int32_t neighbour_count = 1);
  virtual ~LookupEdgesRequest() = default;

  OpRequest* Clone() const override;

  void Init(const Tensor::Map& params) override;
  void Set(const Tensor::Map& tensors) override;

  void Set(const int64_t* edge_ids,
           const int64_t* src_ids,
           int32_t batch_size);

  const std::string& EdgeType() const;
  int32_t Size() const;
  bool Next(int64_t* edge_id, int64_t* src_id);

protected:
  void SetMembers() override;

private:
  int32_t cursor_;
  Tensor* edge_ids_;
  Tensor* src_ids_;
};

class LookupNodesRequest : public OpRequest {
public:
  LookupNodesRequest();
  explicit LookupNodesRequest(const std::string& node_type);
  virtual ~LookupNodesRequest() = default;

  OpRequest* Clone() const override;

  void Init(const Tensor::Map& params) override;
  void Set(const Tensor::Map& tensors) override;

  void Set(const int64_t* node_ids, int32_t batch_size);

  const std::string& NodeType() const;
  int32_t Size() const;
  bool Next(int64_t* node_id);

protected:
  void SetMembers() override;

private:
  int32_t cursor_;
  Tensor* node_ids_;
};

class LookupResponse : public OpResponse {
public:
  LookupResponse();
  virtual ~LookupResponse();

  void Swap(OpResponse& right) override;

  void SetSideInfo(const io::SideInfo* info, int32_t batch_size);
  void AppendWeight(float weight);
  void AppendLabel(int32_t label);
  void AppendAttribute(const io::AttributeValue* value);

  int32_t Size() const { return batch_size_; }
  int32_t Format() const;
  int32_t IntAttrNum() const;
  int32_t FloatAttrNum() const;
  int32_t StringAttrNum() const;
  const float* Weights() const;
  const int32_t* Labels() const;
  const int64_t* IntAttrs() const;
  const float* FloatAttrs() const;
  const std::string* const* StringAttrs() const;

protected:
  void SetMembers() override;

protected:
  io::SideInfo* info_;
  Tensor* infos_;
  Tensor* weights_;
  Tensor* labels_;
  Tensor* i_attrs_;
  Tensor* f_attrs_;
  Tensor* s_attrs_;
};

class LookupEdgesResponse : public LookupResponse {
public:
  LookupEdgesResponse();
  virtual ~LookupEdgesResponse() = default;

  OpResponse* New() const override {
    return new LookupEdgesResponse;
  }
};

class LookupNodesResponse : public LookupResponse {
public:
  LookupNodesResponse();
  virtual ~LookupNodesResponse() = default;

  OpResponse* New() const override {
    return new LookupNodesResponse;
  }
};

class GetCountRequest : public OpRequest {
public:
  GetCountRequest();
  virtual ~GetCountRequest() = default;
};

class GetCountResponse : public OpResponse {
public:
  GetCountResponse();
  virtual ~GetCountResponse() = default;

  OpResponse* New() const override {
    return new GetCountResponse;
  }

  void Swap(OpResponse& right) override;
  void Init(int32_t type_num);
  void Append(int32_t count);
  const int32_t* Count() const;

protected:
  void SetMembers() override;

private:
  Tensor* count_;
};

class GetDegreeRequest : public OpRequest {
public:
  GetDegreeRequest();
  GetDegreeRequest(const std::string& edge_type,
                   NodeFrom node_from);
  virtual ~GetDegreeRequest() = default;

  OpRequest* Clone() const override;

  void Init(const Tensor::Map& params) override;
  void Set(const Tensor::Map& tensors) override;
  void Set(const int64_t* node_ids, int32_t batch_size);

  const std::string& EdgeType() const;
  NodeFrom GetNodeFrom() const;
  const int64_t* GetNodeIds() const;
  int32_t BatchSize() const;

protected:
  void SetMembers() override;

private:
  Tensor* node_ids_;
};

class GetDegreeResponse : public OpResponse {
public:
  GetDegreeResponse();
  virtual ~GetDegreeResponse() = default;

  OpResponse* New() const override {
    return new GetDegreeResponse;
  }

  void Swap(OpResponse& right) override;

  void InitDegrees(int32_t count);
  void AppendDegree(int32_t degree);
  int32_t Size() const { return batch_size_; }

  int32_t* GetDegrees();

protected:
  void SetMembers() override;

private:
  Tensor* degrees_;
};


class GetStatsRequest : public OpRequest {
public:
  GetStatsRequest();
  virtual ~GetStatsRequest() = default;
};

class GetStatsResponse : public OpResponse {
public:
  GetStatsResponse();
  virtual ~GetStatsResponse() = default;

  OpResponse* New() const override {
    return new GetStatsResponse;
  }
  void Swap(OpResponse& right) override;
  void SetCounts(const Counts& counts);
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_GRAPH_REQUEST_H_
