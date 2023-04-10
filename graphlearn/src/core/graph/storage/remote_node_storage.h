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

#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_REMOTE_NODE_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_REMOTE_NODE_STORAGE_H_

#include "core/graph/storage/node_cache.h"
#include "core/io/element_value.h"
#include "include/graph_request.h"
#include "include/config.h"

namespace graphlearn {
namespace io {

class RemoteNodeStorage {
public:
  explicit RemoteNodeStorage();
  ~RemoteNodeStorage();

  Status LookupNodes(int32_t remote_server_id,
                      const LookupNodesRequest* req,
                      LookupNodesResponse* res);

  void SetSideInfo(const SideInfo* info);
  const SideInfo* GetSideInfo() const;

private:
  Status BuildLocalCache(int32_t remote_server_id,
    const std::vector<int64_t>& remaining_ids,
    const std::string& node_type,
    std::map<int64_t, NodeValue>& remote_node_values);

  void ParseResponseValue(NodeValue& value,
    std::shared_ptr<LookupNodesResponse>& res,
    int cursor);

  Status BuildResponse(const LookupNodesRequest* req,
    LookupNodesResponse* res,
    const std::vector<int64_t>& total_ids,
    const std::map<int64_t, NodeValue>& local_node_values,
    const std::map<int64_t, NodeValue>& remote_node_values);

  Status LookupRemoteNodes(int32_t remote_server_id,
    const LookupNodesRequest* req,
    LookupNodesResponse* res);

private:
  NodeCache<int64_t, NodeValue, LFUCachePolicy>* local_cache_;
  SideInfo side_info_;
  bool enable_local_cache_;
};

}  // namespace io
}  // namespace graphlearn


#endif // GRAPHLEARN_CORE_GRAPH_STORAGE_REMOTE_NODE_STORAGE_H_