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

#ifndef GRAPHLEARN_SERVICE_SERVER_IMPL_H_
#define GRAPHLEARN_SERVICE_SERVER_IMPL_H_

#include <memory>
#include <string>
#include <vector>
#include "include/data_source.h"
#include "include/graph_request.h"

namespace graphlearn {

class Env;
class Executor;
class GraphStore;
class InMemoryService;
class DistributeService;
class Coordinator;

class ServerImpl {
public:
  ServerImpl(int32_t server_id,
             int32_t server_count,
             const std::string& server_host,
             const std::string& tracker);
  virtual ~ServerImpl();

  virtual void Start() = 0;
  virtual void Init(const std::vector<io::EdgeSource>& edges,
                    const std::vector<io::NodeSource>& nodes) = 0;
  virtual void Stop() = 0;
  virtual void StopSampling() = 0;
  virtual const Counts& GetStats() const = 0;

protected:
  void RegisterBasicService(Env* env, Executor* executor);
  void InitBasicService();
  void BuildBasicService();
  void StopBasicService();
  void StopDagService();

protected:
  int32_t            server_id_;
  int32_t            server_count_;
  std::string        server_host_;
  InMemoryService*   in_memory_service_;
  DistributeService* dist_service_;
  Coordinator*       coordinator_;
};

class DefaultServerImpl : public ServerImpl {
public:
  DefaultServerImpl(int32_t server_id,
                    int32_t server_count,
                    const std::string& server_host,
                    const std::string& tracker);
  ~DefaultServerImpl() override;

  void Start() override;
  void Init(const std::vector<io::EdgeSource>& edges,
            const std::vector<io::NodeSource>& nodes) override;
  void Stop() override;
  void StopSampling() override;
  const Counts& GetStats() const override;

private:
  Env*        env_;
  GraphStore* graph_store_;
  Executor*   executor_;
};

ServerImpl* NewDefaultServerImpl(int32_t server_id,
                                 int32_t server_count,
                                 const std::string& server_host,
                                 const std::string& tracker);
ServerImpl* NewActorServerImpl(int32_t server_id,
                               int32_t server_count,
                               const std::string& server_host,
                               const std::string& tracker);

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_SERVER_IMPL_H_
