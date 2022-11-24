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

#ifndef GRAPHLEARN_INCLUDE_DAG_REQUEST_H_
#define GRAPHLEARN_INCLUDE_DAG_REQUEST_H_

#include <string>
#include <unordered_map>
#include "include/config.h"
#include "include/op_request.h"
#include "include/tensor.h"
#include "generated/proto/dag.pb.h"

namespace graphlearn {

class Tape;

class DagRequest : public BaseRequest {
public:
  DagRequest();
  void SerializeTo(void* request) override;
  bool ParseFrom(const void* request) override;
  bool ParseFrom(const void* request, const bool copy=false);

  std::string Name() const override;

public:
  DagDef def_;
};

class GetDagValuesRequest : public BaseRequest {
public:
  GetDagValuesRequest();
  GetDagValuesRequest(int32_t dag_id,
                      int32_t client_id = GLOBAL_FLAG(ClientId));

  std::string Name() const override {
    return "GetDagValuesRequest";
  }

  int32_t Id() const {
    return id_;
  }

  int32_t ClientId() const {
    return client_id_;
  }

  void SerializeTo(void* request) override;
  bool ParseFrom(const void* request) override;

private:
  int32_t id_;
  int32_t client_id_;
};

class GetDagValuesResponse : public BaseResponse {
public:
  GetDagValuesResponse();
  GetDagValuesResponse(GetDagValuesResponse&& res);
  ~GetDagValuesResponse() = default;

  void MoveFrom(Tape* tape);

  /// Get the Tensor with given key from the result of each DagNode.
  Tensor* GetValue(int32_t node_id, const std::string& key);

  void SetIndex(int32_t index);
  int32_t Index() { return index_; }

  void SetEpoch(int32_t epoch);
  int32_t Epoch() { return epoch_; }

  bool Valid() { return records_.size() > 0; }

  void SerializeTo(void* response) override;
  bool ParseFrom(const void* response) override;

private:
  int32_t epoch_;
  int32_t index_;

public:
  std::unordered_map<int32_t, Tensor::Map> records_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_DAG_REQUEST_H_
