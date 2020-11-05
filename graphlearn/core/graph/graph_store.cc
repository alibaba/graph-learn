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

#include "graphlearn/core/graph/graph_store.h"

#include <memory>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/common/base/progress.h"
#include "graphlearn/common/threading/sync/cond.h"
#include "graphlearn/core/graph/storage/storage_mode.h"
#include "graphlearn/core/io/element_value.h"
#include "graphlearn/core/io/edge_loader.h"
#include "graphlearn/core/io/node_loader.h"
#include "graphlearn/core/operator/operator_factory.h"
#include "graphlearn/core/runner/distribute_runner.h"
#include "graphlearn/include/config.h"
#include "graphlearn/include/graph_request.h"
#include "graphlearn/platform/env.h"

namespace graphlearn {

namespace {

PROGRESSING(LoadGraphEdges);
PROGRESSING(LoadGraphNodes);

struct LoadGraphEdges {
  void Update(int32_t n) {
    UPDATE_PROGRESSING(LoadGraphEdges, n);
  }
};

struct LoadGraphNodes {
  void Update(int32_t n) {
    UPDATE_PROGRESSING(LoadGraphNodes, n);
  }
};

template<typename Source,
         typename Loader,
         typename DataType,
         typename ReqType,
         typename ResType,
         typename ProgressType>
class Initializer {
public:
  explicit Initializer(Env* env) : env_(env) {
    thread_num_ = GLOBAL_FLAG(InterThreadNum) / 2;
    if (thread_num_ < 1) {
      thread_num_ = 1;
    }
    loaders_.resize(thread_num_, nullptr);
  }

  ~Initializer() {
    for (size_t i = 0; i < loaders_.size(); ++i) {
      delete loaders_[i];
    }
  }

  Status Run(const std::vector<Source>& sources) {
    SyncVariable sv(thread_num_);
    std::vector<Status> s(thread_num_);
    for (int32_t i = 0; i < thread_num_; ++i) {
      loaders_[i] = new Loader(sources, env_, i, thread_num_);
      Closure<void>* task = NewClosure(
        this,
        &Initializer<Source,
                     Loader,
                     DataType,
                     ReqType,
                     ResType,
                     ProgressType>::RunInThread,
        loaders_[i], &(s[i]), &sv);
      env_->InterThreadPool()->AddTask(task);
    }
    sv.Wait();
    return error::FirstErrorIfFound(s);
  }

private:
  void RunInThread(Loader* loader,
                   Status* ret,
                   SyncVariable* sv) {
    int32_t batch_size = GLOBAL_FLAG(DataInitBatchSize);

    bool move_to_next = false;
    Status s = loader->BeginNextFile();
    while (s.ok()) {
      const io::SideInfo* info = loader->GetSideInfo();

      std::unique_ptr<ReqType> req(new ReqType(info, batch_size));
      s = FillRequest(loader, batch_size, req.get());
      if (s.ok()) {
        s = Update(req.get());
      } else if (error::IsOutOfRange(s)) {
        s = Update(req.get());
        move_to_next = true;
      } else {
        break;
      }
      progress_.Update(req->Size());

      if (move_to_next) {
        s = loader->BeginNextFile();
        move_to_next = false;
      }
    }

    if (error::IsOutOfRange(s)) {
      *ret = Status::OK();
    } else {
      *ret = s;
    }

    sv->Inc();
  }

  Status FillRequest(Loader* loader, int32_t n, ReqType* req) {
    Status s;
    DataType value;
    for (int32_t i = 0; i < n; ++i) {
      s = loader->Read(&value);
      if (s.ok()) {
        req->Append(&value);
      } else {
        break;
      }
    }
    return s;
  }

  Status Update(ReqType* req) {
    if (req->Size() <= 0) {
      return Status::OK();
    }

    std::unique_ptr<ResType> res(new ResType);
    op::Operator* op =
      op::OperatorFactory::GetInstance().Lookup(req->Name());
    std::unique_ptr<OpRunner> runner = GetOpRunner(env_, op);
    return runner->Run(req, res.get());
  }

private:
  Env* env_;
  int32_t thread_num_;
  std::vector<Loader*> loaders_;
  ProgressType progress_;
};

}  // anonymous namespace

GraphStore::GraphStore(Env* env)
    : env_(env), graphs_(nullptr), noders_(nullptr) {
  graphs_ = new HeterDispatcher<Graph>(CreateRemoteGraph);
  noders_ = new HeterDispatcher<Noder>(CreateRemoteNoder);
}

GraphStore::~GraphStore() {
  delete graphs_;
  delete noders_;
}

Status GraphStore::Load(
    const std::vector<io::EdgeSource>& edges,
    const std::vector<io::NodeSource>& nodes) {
  for (const auto& e : edges) {
    topo_.Add(e.edge_type, e.src_id_type, e.dst_id_type);
    graphs_->LookupOrCreate(e.edge_type);
  }
  for (const auto& n : nodes) {
    noders_->LookupOrCreate(n.id_type);
  }

  if (io::IsVineyardStorageEnabled()) {
    return Status::OK();
  }

  Initializer<io::EdgeSource,
              io::EdgeLoader,
              io::EdgeValue,
              UpdateEdgesRequest,
              UpdateEdgesResponse,
              LoadGraphEdges> edge_initializer(env_);
  Status s = edge_initializer.Run(edges);
  if (!s.ok()) {
    LOG(ERROR) << "Load graph edges failed, " << s.ToString();
    USER_LOG("Load graph edges failed.");
    USER_LOG(s.ToString());
    return s;
  } else {
    LOG(INFO) << "Load graph edges succeed.";
  }

  Initializer<io::NodeSource,
              io::NodeLoader,
              io::NodeValue,
              UpdateNodesRequest,
              UpdateNodesResponse,
              LoadGraphNodes> node_initializer(env_);
  s = node_initializer.Run(nodes);
  if (!s.ok()) {
    LOG(ERROR) << "Load graph nodes failed, " << s.ToString();
    USER_LOG("Load graph nodes failed.");
    USER_LOG(s.ToString());
  } else {
    LOG(INFO) << "Load graph nodes succeed.";
  }
  return s;
}

void GraphStore::Build() {
  std::string type;

  graphs_->ResetNext();
  Graph* graph = nullptr;
  while (graphs_->Next(&type, &graph)) {
    graph->Build();
  }

  noders_->ResetNext();
  Noder* noder = nullptr;
  while (noders_->Next(&type, &noder)) {
    noder->Build();
  }
  LOG(INFO) << "GraphStore build OK.";
}

Graph* GraphStore::GetGraph(const std::string& edge_type) {
  return graphs_->LookupOrCreate(edge_type);
}

Noder* GraphStore::GetNoder(const std::string& node_type) {
  return noders_->LookupOrCreate(node_type);
}

const Topology& GraphStore::GetTopology() const {
  return topo_;
}

}  // namespace graphlearn
