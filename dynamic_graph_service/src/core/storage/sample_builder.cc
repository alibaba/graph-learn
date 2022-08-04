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

#include "core/storage/sample_builder.h"

#include "core/io/record_builder.h"

namespace dgs {
namespace storage {

SampleBuilder::SampleBuilder(const std::vector<PartitionId>& pids,
                             Partitioner&& partitioner)
  : sampler_factory_(&(SamplerFactory::GetInstance())) {
  for (auto pid : pids) {
    AddPartitionedTable(pid, false);
  }
  SetPartitioner(std::move(partitioner));
}

void SampleBuilder::Init(const execution::Dag* dag) {
  static bool is_inited = false;
  std::lock_guard<std::mutex> guard(builder_mtx_);
  if (is_inited) {
    LOG(INFO) << "SampleBuilder is already inited.";
    return;
  }

  for (auto* node : dag->nodes()) {
    if (node->kind() == PlanNode::Kind_EDGE_SAMPLER) {
      AddEdgeSamplerParams(node->GetParamMap(), node->id());
    } else if (node->kind() == PlanNode::Kind_VERTEX_SAMPLER) {
      AddVertexSamplerParams(node->GetParamMap(), node->id());
    }
  }

  is_inited = true;
  LOG(INFO) << "SampleBuilder is inited by query plan.";
}

void SampleBuilder::AddEdgeSamplerParams(const ParamMap& params,
                                         OperatorId opid) {
  EdgeType etype = params.at("etype");
  SamplerType stype = static_cast<SamplerType>(params.at("strategy"));
  Capacity cap = params.at("fanout");
  eop_table_.emplace(opid, SampleOpInfo{stype, cap});
  etype_ops_table_[etype].push_back(opid);
}

void SampleBuilder::AddVertexSamplerParams(const ParamMap& params,
                                           OperatorId opid) {
  VertexType vtype = params.at("vtype");
  Capacity cap = params.at("versions");
  vop_table_.emplace(opid, cap);
  vtype_ops_table_.emplace(vtype, opid);
}

VertexSamplerPtr& SampleBuilder::GetVertexSampler(
    const Key::Prefix& prefix_key) {
  auto pid = partitioner_.GetPartitionId(prefix_key.vid);
  auto &vertex_table = table_insts_[pid].vtable;
  auto iter = vertex_table.find(prefix_key);

  if (__builtin_expect(iter == vertex_table.end(), false)) {
    // create if not exist.
    auto capacity = vop_table_[prefix_key.op_id];
    vertex_table.emplace(prefix_key, std::make_unique<VertexSampler>(capacity));
    iter = vertex_table.find(prefix_key);
  }
  return iter->second;
}

EdgeSamplerPtr& SampleBuilder::GetEdgeSampler(const Key::Prefix& prefix_key) {
  auto pid = partitioner_.GetPartitionId(prefix_key.vid);
  auto &edge_table = table_insts_[pid].etable;
  auto iter = edge_table.find(prefix_key);
  if (__builtin_expect(iter == edge_table.end(), false)) {
    // create if not exist.
    auto &op_info = eop_table_[prefix_key.op_id];
    auto sampler = sampler_factory_->CreateEdgeSampler(
      op_info.stype, op_info.capacity);
    edge_table.emplace(prefix_key, std::move(sampler));
    iter = edge_table.find(prefix_key);
  }
  return iter->second;
}

void SampleBuilder::SetPartitioner(Partitioner&& partitioner) {
  LOG(INFO) << "New partitioner is set.";
  partitioner_ = std::move(partitioner);
}

void SampleBuilder::AddPartitionedTable(PartitionId pid, bool is_restore) {
  std::lock_guard<std::mutex> guard(builder_mtx_);
  if (pid < table_insts_.size() && table_insts_[pid].is_open) {
    LOG(WARNING) << "SampleBuilder Table Partition "
                 << pid << " already exists.";
    return;
  }

  if (pid >= table_insts_.size()) {
    // expand vector size.
    table_insts_.resize(pid + 1);
  }

  table_insts_[pid].is_open = true;
  table_insts_[pid].vtable = PartitionedTable::VTable();
  table_insts_[pid].etable = PartitionedTable::ETable();

  LOG(INFO) << "SampleBuilder Table Partition " << pid << " is added.";
}

void SampleBuilder::RemovePartitionedTable(PartitionId pid) {
  std::lock_guard<std::mutex> guard(builder_mtx_);
  if (pid >= table_insts_.size() || !table_insts_[pid].is_open) {
    LOG(WARNING) << "SampleBuilder Table Partition "
                 << pid << " doesn't exist.";
    return;
  }

  table_insts_[pid].is_open = false;
  table_insts_[pid].vtable = PartitionedTable::VTable();
  table_insts_[pid].etable = PartitionedTable::ETable();

  LOG(INFO) << "SampleBuilder Table Partition " << pid << " is removed.";
}

std::vector<KVPair> SampleBuilder::Sample(const io::RecordBatch& batch) {
  std::vector<KVPair> output;
  auto batch_view = batch.GetView();
  for (int i = 0; i < batch_view.RecordNum(); ++i) {
    auto record_view = batch_view.GetRecordByIdx(i);
    if (record_view.Type() == RecordType::VERTEX) {
      auto vtx_record = record_view.AsVertexRecord();
      SampleVertex(vtx_record, &output);
    } else {
      auto edge_record = record_view.AsEdgeRecord();
      SampleEdge(edge_record, &output);
    }
  }
  return output;
}

void SampleBuilder::SampleVertex(
    const io::VertexRecordView& record_view,
    std::vector<KVPair>* output) {
  auto iter = vtype_ops_table_.find(record_view.Type());
  if (iter != vtype_ops_table_.end()) {
    OperatorId op_id = iter->second;
    uint32_t index = 0;
    Key key{record_view.Type(), record_view.Id(), op_id, 0};
    VertexSamplerPtr& sampler = GetVertexSampler(key.pkey);
    auto is_sampled = sampler->Sample(record_view, index);
    if (is_sampled) {
      key.index = index;
      io::RecordBuilder builder;
      builder.BuildFromView(&record_view);

      auto *data = const_cast<char*>(reinterpret_cast<
          const char*>(builder.BufPointer()));
      auto size = builder.BufSize();
      auto buf = act::BytesBuffer(data, size,
          seastar::make_object_deleter(std::move(builder)));

      output->push_back({key, io::Record{std::move(buf)}});
    }
  }
}

void SampleBuilder::SampleEdge(
    const io::EdgeRecordView& record_view,
    std::vector<KVPair>* output) {
  auto iter = etype_ops_table_.find(record_view.Type());
  if (iter != etype_ops_table_.end()) {
    auto &op_ids = iter->second;
    uint32_t index = 0;
    Key key{record_view.SrcType(), record_view.SrcId(), 0, 0};
    for (auto op_id : op_ids) {
      key.pkey.op_id = op_id;
      EdgeSamplerPtr& sampler = GetEdgeSampler(key.pkey);
      auto is_sampled = sampler->Sample(record_view, index);
      if (is_sampled) {
        key.index = index;
        io::RecordBuilder builder;
        builder.BuildFromView(&record_view);

        auto *data = const_cast<char*>(reinterpret_cast<
            const char*>(builder.BufPointer()));
        auto size = builder.BufSize();
        auto buf = act::BytesBuffer(data, size,
            seastar::make_object_deleter(std::move(builder)));

        output->push_back({key, io::Record{std::move(buf)}});
      }
    }
  }
}


void SampleBuilder::Dump(std::ofstream& file) {
  uint64_t num_tables = 0;
  for (auto &table : table_insts_) {
    if (table.is_open) {
      ++num_tables;
    }
  }
  file.write(reinterpret_cast<char*>(&num_tables), sizeof(uint64_t));

  for (auto &table_manager : table_insts_) {
    if (table_manager.is_open) {
      auto &vertex_table = table_manager.vtable;
      uint64_t v_num = vertex_table.size();
      file.write(reinterpret_cast<char*>(&v_num), sizeof(uint64_t));
      auto vertex_iter = vertex_table.begin();
      while (vertex_iter != vertex_table.end()) {
        auto value_buf = vertex_iter->second->Dump();
        uint64_t value_size = value_buf.size();
        file.write(reinterpret_cast<const char*>(&vertex_iter->first),
                   sizeof(Key::Prefix));
        file.write(reinterpret_cast<char*>(&value_size), sizeof(uint64_t));
        file.write(value_buf.get(), value_buf.size());
        vertex_iter++;
      }
    }
  }

  for (auto &table_manager : table_insts_) {
    if (table_manager.is_open) {
      auto& edge_table = table_manager.etable;
      uint64_t e_num = edge_table.size();
      file.write(reinterpret_cast<char*>(&e_num), sizeof(uint64_t));
      auto edge_iter = edge_table.begin();
      while (edge_iter != edge_table.end()) {
        auto value_buf = edge_iter->second->Dump();
        uint64_t value_size = value_buf.size();
        file.write(reinterpret_cast<const char*>(&edge_iter->first),
                   sizeof(Key::Prefix));
        file.write(reinterpret_cast<char*>(&value_size), sizeof(uint64_t));
        file.write(value_buf.get(), value_buf.size());
        edge_iter++;
      }
      file.close();
    }
  }
}

void SampleBuilder::Load(std::ifstream& file) {
  uint64_t num_tables = 0;
  file.read(reinterpret_cast<char*>(&num_tables), sizeof(uint64_t));
  table_insts_ = std::vector<PartitionedTable>(num_tables);

  // FIXME(@goldenleaves): add partition id for PartitionedTable.
  for (int i = 0; i < num_tables; ++i) {
    table_insts_[i].is_open = true;
    auto& vertex_table = table_insts_[i].vtable;
    uint64_t v_num;
    file.read(reinterpret_cast<char*>(&v_num), sizeof(uint64_t));
    for (int i = 0; i < v_num; i++) {
      Key::Prefix prefix_key{0, 0, 0};
      uint64_t value_size;
      file.read(reinterpret_cast<char*>(&prefix_key), sizeof(Key::Prefix));
      file.read(reinterpret_cast<char*>(&value_size), sizeof(uint64_t));
      act::BytesBuffer value_buf(value_size);
      file.read(value_buf.get_write(), value_size);

      auto capacity = vop_table_[prefix_key.op_id];
      auto sampler = std::make_unique<VertexSampler>(capacity);
      sampler->Load(value_buf);
      vertex_table.emplace(prefix_key, std::move(sampler));
    }
  }

  for (int i = 0; i < num_tables; ++i) {
    table_insts_[i].is_open = true;
    auto& edge_table = table_insts_[i].etable;
    uint64_t e_num;
    file.read(reinterpret_cast<char*>(&e_num), sizeof(uint64_t));
    for (int i = 0; i < e_num; i++) {
      Key::Prefix prefix_key{0, 0, 0};
      uint64_t value_size;
      file.read(reinterpret_cast<char*>(&prefix_key), sizeof(Key::Prefix));
      file.read(reinterpret_cast<char*>(&value_size), sizeof(uint64_t));
      act::BytesBuffer value_buf(value_size);
      file.read(value_buf.get_write(), value_size);

      auto op_info = eop_table_[prefix_key.op_id];
      auto sampler = sampler_factory_->CreateEdgeSampler(
        op_info.stype, op_info.capacity);
      sampler->Load(value_buf);
      edge_table.emplace(prefix_key, std::move(sampler));
    }
  }

  file.close();
}

}  // namespace storage
}  // namespace dgs
