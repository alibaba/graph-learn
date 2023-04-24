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

#include "core/graph/storage/remote_node_storage.h"

#include <memory>
#include "common/base/log.h"
#include "include/client.h"
#include "include/config.h"
#include "core/io/element_value.h"

namespace graphlearn {
namespace io {
RemoteNodeStorage::RemoteNodeStorage() {
  int32_t local_node_cache_capacity = GLOBAL_FLAG(LocalNodeCacheCapacity);
  enable_local_cache_ = local_node_cache_capacity > 0;
  if(enable_local_cache_) {
    LOG(INFO) << "local node cache enabled by capacity = : " << local_node_cache_capacity;
    local_cache_ = new NodeCache<int64_t, NodeValue, LFUCachePolicy>(
      local_node_cache_capacity, LFUCachePolicy<int64_t>());
  } else {
    LOG(INFO) << "local node cache disabled";
  }
}

RemoteNodeStorage::~RemoteNodeStorage() {
  if(enable_local_cache_)
    delete local_cache_;
}

void RemoteNodeStorage::SetSideInfo(const SideInfo* info)
{
  if (!side_info_.IsInitialized()) {
    side_info_.CopyFrom(*info);
  }
}

const SideInfo* RemoteNodeStorage::GetSideInfo() const
{
  return &side_info_;
}

Status RemoteNodeStorage::LookupNodes(int32_t remote_server_id,
  const LookupNodesRequest* req,
  LookupNodesResponse* res){
  assert(req != nullptr && res != nullptr);
  if(!enable_local_cache_) {
    return LookupRemoteNodes(remote_server_id, req, res);
  }
  res->SetSideInfo(&side_info_, req->Size());
  int64_t node_id = -1;
  std::vector<int64_t> remaining_ids;
  std::vector<int64_t> total_ids;
  std::map<int64_t, NodeValue> cached_node_values;
  while (req->Next(&node_id)) {
    total_ids.push_back(node_id);
    NodeValue value;
    if(local_cache_->TryGet(node_id, value)) {
      cached_node_values.emplace(node_id, std::move(value));
    } else {
        remaining_ids.push_back(node_id);
    }
  }

  std::map<int64_t, NodeValue> remote_node_values;
  Status ret = BuildLocalCache(remote_server_id, remaining_ids, req->NodeType(), remote_node_values);
  if (!ret.ok()) return ret;

  return BuildResponse(req, res, total_ids,  cached_node_values, remote_node_values);
}

Status RemoteNodeStorage::LookupRemoteNodes(int32_t remote_server_id,
    const LookupNodesRequest* req,
    LookupNodesResponse* res) {
  std::unique_ptr<Client> client(NewRpcClient(remote_server_id));
  return client->LookupNodes(req, res);
}

Status RemoteNodeStorage::BuildLocalCache(int32_t remote_server_id,
  const std::vector<int64_t>& remaining_ids,
  const std::string& node_type,
  std::map<int64_t, NodeValue>& remote_node_values) {
  std::shared_ptr<LookupNodesRequest> partial_req(new LookupNodesRequest(node_type));
  partial_req->Set(remaining_ids.data(), remaining_ids.size());
  partial_req->DisableShard();
  std::shared_ptr<LookupNodesResponse> partial_res(new LookupNodesResponse());
  Status ret = LookupRemoteNodes(remote_server_id, partial_req.get(), partial_res.get());
  if(!ret.ok()) return ret;

  for(int index = 0; index < remaining_ids.size(); index++) {
    int64_t node_id = remaining_ids[index];
    NodeValue value;
    value.id = node_id;
    ParseResponseValue(value, partial_res, index);
    local_cache_->Insert(node_id, value);
    remote_node_values.emplace(node_id, std::move(value));
  }

  return ret;
}

void RemoteNodeStorage::ParseResponseValue(NodeValue& value,
  std::shared_ptr<LookupNodesResponse>& res,
  int cursor) {
  if (side_info_.IsWeighted())
    value.weight = res->Weights()[cursor];
  if (side_info_.IsLabeled()) 
    value.label = res->Labels()[cursor];
  if (side_info_.IsTimestamped())
    value.timestamp = res->Timestamps()[cursor];
  value.attrs->Reserve(side_info_.i_num, side_info_.f_num, side_info_.s_num);
  if (side_info_.i_num > 0)
    value.attrs->Add(res->IntAttrs() + (cursor*side_info_.i_num), side_info_.i_num);
  if (side_info_.f_num > 0)
    value.attrs->Add(res->FloatAttrs()+ (cursor*side_info_.f_num), side_info_.f_num);

  for (int32_t i = 0; i < side_info_.s_num; ++i) {
    value.attrs->Add(res->StringAttrs()[cursor*side_info_.s_num][i]);
  }
}

Status RemoteNodeStorage::BuildResponse(const LookupNodesRequest* req,
  LookupNodesResponse* res,
  const std::vector<int64_t>& total_ids,
  const std::map<int64_t, NodeValue>& local_node_values,
  const std::map<int64_t, NodeValue>& remote_node_values) {
    int64_t node_id = 0;
    for(int index = 0; index < total_ids.size(); index++) {
      node_id = total_ids[index];
      const NodeValue* value = nullptr;
      std::map<int64_t, NodeValue>::const_iterator node_iter;
      node_iter = local_node_values.find(node_id);
      if( node_iter != local_node_values.end()) {
        value = & (node_iter->second);
      } else {
        node_iter = remote_node_values.find(node_id);
        if (node_iter == remote_node_values.end())
          return Status(error::NOT_FOUND, "failed to fetch node attribute.");
        value =& (node_iter->second);
      }

      if (side_info_.IsWeighted())
        res->AppendWeight(value->weight);
      if (side_info_.IsLabeled())
        res->AppendLabel(value->label);
      if (side_info_.IsTimestamped())
        res->AppendTimestamp(value->timestamp);
      res->AppendAttribute(value->attrs);
    }

    return Status::OK();
  }

}  // namespace io
}  // namespace graphlearn