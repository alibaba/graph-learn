/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef DGS_CORE_STORAGE_SAMPLE_BUILDER_H_
#define DGS_CORE_STORAGE_SAMPLE_BUILDER_H_

#include <fstream>

#include "core/execution/dag.h"
#include "core/storage/sampler_factory.h"
#include "core/storage/sample_store.h"

namespace dgs {
namespace storage {

class SampleBuilder {
public:
  SampleBuilder(const std::vector<PartitionId>& pids,
                Partitioner&& partitioner);
  ~SampleBuilder() = default;

  void Init(const execution::Dag* dag);

  std::vector<KVPair> Sample(const io::RecordBatch& batch);

  void Dump(std::ofstream& file);
  void Load(std::ifstream& file);

  void AddEdgeSamplerParams(const ParamMap& params,
                            OperatorId opid);
  void AddVertexSamplerParams(const ParamMap& params,
                              OperatorId opid);

private:
  void SampleVertex(const io::VertexRecordView& record,
                    std::vector<KVPair>* output);
  void SampleEdge(const io::EdgeRecordView& record,
                  std::vector<KVPair>* output);

  VertexSamplerPtr& GetVertexSampler(const Key::Prefix& key);
  EdgeSamplerPtr& GetEdgeSampler(const Key::Prefix& key);

  void SetPartitioner(Partitioner&& partitioner);
  void AddPartitionedTable(PartitionId pid, bool is_restore);
  void RemovePartitionedTable(PartitionId pid);

private:
  struct Hasher {
    std::size_t operator() (const Key::Prefix &p) const {
      std::size_t h1 = std::hash<VertexType>()(p.vtype);
      std::size_t h2 = std::hash<VertexId>()(p.vid);
      std::size_t h3 = std::hash<OperatorId>()(p.op_id);
      return h1 ^ h2 ^ h3;
    }
  };

  struct SampleOpInfo {
    SamplerType stype;
    Capacity capacity;
  };

  struct PartitionedTable {
    using VTable = std::unordered_map<Key::Prefix, VertexSamplerPtr, Hasher>;
    using ETable = std::unordered_map<Key::Prefix, EdgeSamplerPtr, Hasher>;

    VTable vtable;
    ETable etable;
    bool   is_open{false};
  };

private:
  std::vector<PartitionedTable> table_insts_;
  std::mutex                    builder_mtx_;
  Partitioner                   partitioner_;
  SamplerFactory*               sampler_factory_;
  std::unordered_map<EdgeType, std::vector<OperatorId>> etype_ops_table_;
  std::unordered_map<VertexType, OperatorId>            vtype_ops_table_;
  std::unordered_map<OperatorId, SampleOpInfo>          eop_table_;
  std::unordered_map<OperatorId, Capacity>              vop_table_;
};

}  // namespace storage
}  // namespace dgs

#endif  // DGS_CORE_STORAGE_SAMPLE_BUILDER_H_
