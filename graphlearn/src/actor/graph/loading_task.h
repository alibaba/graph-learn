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

#ifndef GRAPHLEARN_ACTOR_GRAPH_PARSING_TASK_H_
#define GRAPHLEARN_ACTOR_GRAPH_PARSING_TASK_H_

#include <memory>
#include <vector>
#include "brane/core/alien_thread_pool.hh"
#include "actor/graph/output_handle.h"
#include "actor/graph/wrapper_source.h"
#include "common/io/value.h"
#include "core/io/edge_loader.h"
#include "core/io/node_loader.h"
#include "include/data_source.h"

namespace graphlearn {
namespace actor {

class NodeLoadingTask : public brane::alien_task {
public:
  NodeLoadingTask(io::NodeLoader* reader,
                  unsigned id,
                  unsigned batch_size);
  ~NodeLoadingTask() override = default;
  void run() override;

private:
  inline Status BeginNextFile() {
    Status s = reader_->BeginNextFile(&source_);
    source_wrapper_ = SourceWrapper<io::NodeSource>(source_);
    return s;
  }

private:
  io::NodeLoader*               reader_;
  io::NodeSource*               source_;
  unsigned                      batch_size_;
  io::NodeValue                 value_;
  NodeOutputHandle              output_handle_;
  SourceWrapper<io::NodeSource> source_wrapper_;
};

class EdgeLoadingTask : public brane::alien_task {
public:
  EdgeLoadingTask(io::EdgeLoader* reader,
                  unsigned id,
                  unsigned batch_size);
  ~EdgeLoadingTask() override = default;
  void run() override;

private:
  inline Status BeginNextFile() {
    Status s = reader_->BeginNextFile(&source_);
    source_wrapper_ = SourceWrapper<io::EdgeSource>(source_);
    return s;
  }

private:
  io::EdgeLoader*               reader_;
  io::EdgeSource*               source_;
  unsigned                      batch_size_;
  io::EdgeValue                 value_;
  EdgeOutputHandle              output_handle_;
  SourceWrapper<io::EdgeSource> source_wrapper_;
};

}  // namespace actor
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_GRAPH_PARSING_TASK_H_
