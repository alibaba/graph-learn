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

#ifndef GRAPHLEARN_INCLUDE_OP_REQUEST_H_
#define GRAPHLEARN_INCLUDE_OP_REQUEST_H_

#include <cstdint>
#include <memory>
#include <mutex>  // NOLINT [build/c++11]
#include <string>
#include <unordered_map>
#include "include/constants.h"
#include "include/request.h"
#include "include/sparse_tensor.h"
#include "include/status.h"
#include "include/tensor.h"

namespace graphlearn {

class OpRequest : public ShardableRequest<OpRequest> {
public:
  explicit OpRequest(const std::string& shard_key=kUnspecified);
  virtual ~OpRequest() = default;

  virtual void Init(const Tensor::Map& params) {}
  virtual void Set(const Tensor::Map& tensors,
                   const SparseTensor::Map& sparse_tensor={}) {}

  std::string Name() const override;

  virtual OpRequest* Clone() const;

  void SerializeTo(void* request) override;
  bool ParseFrom(const void* request) override;

  ShardsPtr<OpRequest> Partition() const override;

protected:
  // All parameters and values come from params_ and tensors_.
  // To simplify usage, we may need some private members pointing to
  // the elements of params_ and tensors_, such as
  // ``` src_ids_ = &(params_[kNeighborCount]); ```
  // ``` Finalize() ``` is designed for this. Requests inheriting from
  // ``` OpRequest ``` should implement their own ``` Finalize() ```.
  virtual void Finalize() {}
  // virtual void Finalize() {} // TODO:

public:
  std::unordered_map<std::string, Tensor> params_;
  std::unordered_map<std::string, Tensor> tensors_;
  std::unordered_map<std::string, SparseTensor> sparse_tensors_;

protected:
  bool is_parse_from_;
};

class OpResponse : public JoinableResponse<OpResponse> {
public:
  OpResponse();
  virtual ~OpResponse() = default;

  virtual OpResponse* New() const {
    return new OpResponse;
  }

  void SerializeTo(void* response) override;
  bool ParseFrom(const void* response) override;

  void Stitch(ShardsPtr<OpResponse> shards) override;
  virtual void Swap(OpResponse& right);

  virtual bool IsFinalized() const { return true;}

protected:
  // All parameters and values come from params_ and tensors_.
  // To simplify usage, we may need some private members pointing to
  // the elements of params_ and tensors_, such as
  // ``` src_ids_ = &(params_[kNeighborCount]); ```
  // ``` Finalize() ``` is designed for this. Responses inheriting from
  // ``` OpResponse ``` should implement their own ``` Finalize() ```.
  // virtual void Finalize() {}
  virtual void Finalize() {

  }

public:
  int32_t batch_size_;
  std::unordered_map<std::string, Tensor> params_;
  std::unordered_map<std::string, Tensor> tensors_;
  std::unordered_map<std::string, SparseTensor> sparse_tensors_;

protected:
  bool is_parse_from_;
};

typedef std::unique_ptr<OpRequest> OpRequestPtr;
typedef std::unique_ptr<OpResponse> OpResponsePtr;

class RequestFactory {
public:
  typedef OpRequest* (*RequestCreator)();
  typedef OpResponse* (*ResponseCreator)();

  static RequestFactory* GetInstance() {
    static RequestFactory factory;
    return &factory;
  }

  void Register(const std::string&, RequestCreator, ResponseCreator);
  OpRequest* NewRequest(const std::string& name);
  OpResponse* NewResponse(const std::string& name);

private:
  RequestFactory() = default;
  RequestFactory(const RequestFactory&);
  RequestFactory& operator=(const RequestFactory&);

private:
  std::mutex mtx_;
  std::unordered_map<std::string, RequestCreator> req_;
  std::unordered_map<std::string, ResponseCreator> res_;
};

}  // namespace graphlearn

#define REGISTER_REQUEST(Name, ReqClass, ResClass)                        \
  inline ::graphlearn::OpRequest* New##Name##ReqClass() {                 \
    return new ReqClass();                                                \
  }                                                                       \
  inline ::graphlearn::OpResponse* New##Name##ResClass() {                \
    return new ResClass();                                                \
  }                                                                       \
  class Register##Name##ReqClass {                                        \
  public:                                                                 \
    Register##Name##ReqClass() {                                          \
      auto factory = ::graphlearn::RequestFactory::GetInstance();         \
      factory->Register(#Name, New##Name##ReqClass, New##Name##ResClass); \
    }                                                                     \
  };                                                                      \
  static Register##Name##ReqClass register_##Name##ReqClass;

#endif  // GRAPHLEARN_INCLUDE_OP_REQUEST_H_
