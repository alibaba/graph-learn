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

#ifndef GRAPHLEARN_INCLUDE_REQUEST_H_
#define GRAPHLEARN_INCLUDE_REQUEST_H_

#include <string>
#include "include/shardable.h"
#include "include/config.h"

namespace graphlearn {

class BaseRequest {
public:
  explicit BaseRequest(bool shardable) : shardable_(shardable) {}
  virtual ~BaseRequest() = default;

  bool IsShardable() const { return shardable_; }
  void DisableShard() { shardable_ = false; }

  virtual bool ParseFrom(const void* request) { return true; }
  virtual void SerializeTo(void* request) {}

  virtual std::string Name() const = 0;

protected:
  mutable bool shardable_;
};

class BaseResponse {
public:
  virtual ~BaseResponse() = default;

  virtual bool ParseFrom(const void* response) { return true; }
  virtual void SerializeTo(void* response) {}
};

template <class T>
class ShardableRequest : public BaseRequest, Shardable<T> {
public:
  explicit ShardableRequest(const std::string& shard_key)
    : BaseRequest(true),
      shard_key_(shard_key) {}

  virtual const std::string& ShardKey() const { return shard_key_; }
  virtual int32_t ShardId() const { return GLOBAL_FLAG(ServerId); }
protected:
  const std::string shard_key_;
  virtual ~ShardableRequest() = default;
};

template <class T>
class JoinableResponse : public BaseResponse, Joinable<T> {
public:
  JoinableResponse() {}
  virtual ~JoinableResponse() = default;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_REQUEST_H_
