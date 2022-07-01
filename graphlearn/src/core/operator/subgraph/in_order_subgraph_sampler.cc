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

#include "graphlearn/core/operator/graph/node_generator.h"
#include "graphlearn/core/operator/subgraph/subgraph_sampler.h"

namespace graphlearn {
namespace op {

class InOrderNodeSubGraphSampler : public SubGraphSampler {
public:
  Status SampleSeed(std::set<int64_t>* nodes,
                    GraphStore* graph_store,
                    const std::string& type,
                    int32_t batch_size,
                    int32_t epoch) {
    auto storage = new StorageWrapper(NodeFrom::kNode, type, graph_store);
    std::unique_ptr<Generator> generator;
    generator.reset(new OrderedGenerator(storage));

    if (epoch < generator->Epoch()) {
      return error::OutOfRange("No more nodes exist.");
    }

    ::graphlearn::io::IdType id = 0;
    for (int32_t i = 0; nodes->size() < batch_size; ++i) {
      if (generator->Next(&id)) {
        nodes->insert(id);
      } else {
        break;
      }
    }

    if (nodes->size() > 0) {
      return Status::OK();
    } else {
      // Begin next epoch.
      generator->Reset();
      return error::OutOfRange("No more nodes exist.");
    }
  }
};

REGISTER_OPERATOR("InOrderNodeSubGraphSampler", InOrderNodeSubGraphSampler);

}  // namespace op
}  // namespace graphlearn
