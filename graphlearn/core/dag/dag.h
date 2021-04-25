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

#ifndef GRAPHLEARN_CORE_DAG_DAG_H_
#define GRAPHLEARN_CORE_DAG_DAG_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "graphlearn/core/dag/dag_node.h"
#include "graphlearn/proto/dag.pb.h"

namespace graphlearn {

class Optimizer;

class Dag {
public:
  explicit Dag(const DagDef& dag_def);
  ~Dag() = default;

  int32_t Id() const {
    return id_;
  }

  const std::string& DebugString() const {
    return debug_;
  }

  size_t Size() const {
    return nodes_.size();
  }

  const DagNode* Root() const {
    return root_;
  }

  const std::vector<const DagNode*>& Nodes() const {
    return nodes_;
  }

  void Compile(Optimizer* optimizer);

private:
  friend class Optimizer;

  int32_t        id_;
  std::string    debug_;
  const DagNode* root_;
  std::vector<const DagNode*> nodes_;
};


class DagFactory {
public:
  static DagFactory* GetInstance() {
    static DagFactory factory;
    return &factory;
  }

  ~DagFactory();

  Status Create(const DagDef& def, Dag** dag);
  Dag* Lookup(int32_t dag_id);

private:
  DagFactory() = default;
  DagFactory(const DagFactory&);
  DagFactory& operator=(const DagFactory&);

private:
  std::mutex mtx_;
  std::unordered_map<int32_t, Dag*> map_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_DAG_DAG_H_
