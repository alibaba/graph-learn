/* Copyright 2020-2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_STORAGE_UTILS_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_STORAGE_UTILS_H_

#include <memory>
#include <set>

#if defined(WITH_VINEYARD)
#include "vineyard/graph/fragment/arrow_fragment.vineyard.h"
#endif

#if defined(WITH_VINEYARD)
#include "core/graph/storage/edge_storage.h"
#include "core/graph/storage/graph_storage.h"
#include "core/graph/storage/node_storage.h"
#include "core/graph/storage/topo_storage.h"
#include "include/config.h"

namespace graphlearn {
namespace io {

using vineyard_oid_t = IdType;
using vineyard_vid_t = uint64_t;

using gl_frag_t = vineyard::ArrowFragment<vineyard_oid_t, vineyard_vid_t>;
using vertex_t = gl_frag_t::vertex_t;
using label_id_t = gl_frag_t::label_id_t;

using graphlearn::io::EdgeStorage;
using graphlearn::io::GraphStorage;
using graphlearn::io::NodeStorage;
using graphlearn::io::TopoStorage;

using graphlearn::io::Attribute;
using graphlearn::io::AttributeValue;
using graphlearn::io::EdgeValue;
using graphlearn::io::IdList;
using graphlearn::io::IndexType;
using graphlearn::io::IndexArray;
using graphlearn::io::NewDataHeldAttributeValue;
using graphlearn::io::SideInfo;

void init_table_accessors(const std::shared_ptr<arrow::Table>& table,
                          const std::set<std::string>& attrs,
                          std::vector<int>& i32_indexes,
                          std::vector<int>& i64_indexes,
                          std::vector<int>& f32_indexes,
                          std::vector<int>& f64_indexes,
                          std::vector<int>& s_indexes,
                          std::vector<int>& ls_indexes,
                          std::vector<const void*>& table_accessors);

AttributeValue *arrow_line_to_attribute_value(
                          const int row_index,
                          const std::vector<int>& i32_indexes,
                          const std::vector<int>& i64_indexes,
                          const std::vector<int>& f32_indexes,
                          const std::vector<int>& f64_indexes,
                          const std::vector<int>& s_indexes,
                          const std::vector<int>& ls_indexes,
                          const std::vector<const void*>& table_accessors);

const IdArray get_all_src_ids(const std::shared_ptr<gl_frag_t>& frag,
                              const label_id_t edge_label);

const IdArray get_all_dst_ids(const std::shared_ptr<gl_frag_t>& frag,
                              const label_id_t edge_label);

const IndexArray get_all_in_degree(const std::shared_ptr<gl_frag_t>& frag,
                                   const label_id_t edge_label);

const IndexArray get_all_out_degree(const std::shared_ptr<gl_frag_t>& frag,
                                    const label_id_t edge_label);

const Array<IdType>
get_all_outgoing_neighbor_nodes(const std::shared_ptr<gl_frag_t>& frag,
                                IdType src_id, const label_id_t edge_label);

const Array<IdType>
get_all_outgoing_neighbor_edges(const std::shared_ptr<gl_frag_t>& frag,
                                IdType src_id, const label_id_t edge_label);

const Array<IdType>
get_all_outgoing_neighbor_nodes(const std::shared_ptr<gl_frag_t>& frag,
                                std::vector<IdType> const& edge_lists,
                                IdType src_id, const label_id_t edge_label,
                                const std::vector<std::pair<IdType, IdType>>& edge_offsets_);

const Array<IdType>
get_all_outgoing_neighbor_edges(const std::shared_ptr<gl_frag_t>& frag,
                                const std::vector<IdType>& edge_lists,
                                IdType src_id, const label_id_t edge_label,
                                const std::vector<std::pair<IdType, IdType>>& edge_offsets_);

IdType get_edge_src_id(const std::shared_ptr<gl_frag_t>& frag,
                       label_id_t const edge_label,
                       const std::vector<IdType>& src_ids, IdType edge_id);

IdType get_edge_dst_id(const std::shared_ptr<gl_frag_t>& frag,
                       label_id_t const edge_label,
                       const std::vector<IdType>& dst_ids, IdType edge_id);

float get_edge_weight(const std::shared_ptr<gl_frag_t>& frag,
                      label_id_t const edge_label, IdType edge_offset);

int32_t get_edge_label(const std::shared_ptr<gl_frag_t>& frag,
                       label_id_t const edge_label, IdType edge_offset);

void init_src_dst_list(const std::shared_ptr<gl_frag_t>& frag,
                       label_id_t const edge_label,
                       label_id_t const src_node_label,
                       label_id_t const dst_node_label,
                       std::vector<IdType>& src_lists,
                       std::vector<IdType>& dst_lists,
                       std::vector<IdType>& edge_lists,
                       std::vector<std::pair<IdType, IdType>>& edge_offsets);

SideInfo *frag_edge_side_info(const std::shared_ptr<gl_frag_t>& frag,
                              const std::set<std::string>& attrs,
                              const std::string& edge_label_name,
                              const std::string& src_label_name,
                              const std::string& dst_label_name,
                              label_id_t const edge_label);

SideInfo *frag_node_side_info(const std::shared_ptr<gl_frag_t>& frag,
                              const std::set<std::string>& attrs,
                              const std::string& node_label_name,
                              label_id_t const node_label);

int64_t find_index_of_name(const std::shared_ptr<arrow::Schema>& schema,
                           const std::string& name);

class ArrowRefAttributeValue : public AttributeValue {
public:
  ArrowRefAttributeValue(const int row_index,
                          const std::vector<int>& i32_indexes,
                          const std::vector<int>& i64_indexes,
                          const std::vector<int>& f32_indexes,
                          const std::vector<int>& f64_indexes,
                          const std::vector<int>& s_indexes,
                          const std::vector<int>& ls_indexes,
                          const std::vector<const void*>& table_accessors):
		row_index_(row_index),
		i32_indexes_(i32_indexes),
		i64_indexes_(i64_indexes),
		f32_indexes_(f32_indexes),
		f64_indexes_(f64_indexes),
		s_indexes_(s_indexes),
		ls_indexes_(ls_indexes),
		table_accessors_(table_accessors) {
  }

  ~ArrowRefAttributeValue() override {
  }

  void Reuse(const int row_index) {
    row_index_ = row_index;
  }

  void Clear() override {
  }

  void Shrink() override {
    throw std::runtime_error("Not implemented");
  }

  void Swap(AttributeValue* rhs) override {
    throw std::runtime_error("Not implemented");
  }

  void Reserve(int32_t i_num, int32_t f_num, int32_t s_num) override {
    throw std::runtime_error("Not implemented");
  }

  void Add(int64_t value) override {
    throw std::runtime_error("Not implemented");
  }

  void Add(float value) override {
    throw std::runtime_error("Not implemented");
  }

  void Add(std::string&& value) override {
    throw std::runtime_error("Not implemented");
  }

  void Add(const std::string& value) override {
    throw std::runtime_error("Not implemented");
  }

  void Add(const char* value, int32_t len) override {
    throw std::runtime_error("Not implemented");
  }

  void Add(const int64_t* values, int32_t len) override {
    throw std::runtime_error("Not implemented");
  }

  void Add(const float* values, int32_t len) override {
    throw std::runtime_error("Not implemented");
  }

  const int64_t* GetInts(int32_t* len) const override {
    throw std::runtime_error("Not implemented");
  }

  const float* GetFloats(int32_t* len) const override {
    throw std::runtime_error("Not implemented");
  }

  const std::string* GetStrings(int32_t* len) const override {
    throw std::runtime_error("Not implemented");
  }

  const LiteString* GetLiteStrings(int32_t* len) const override {
    throw std::runtime_error("Not implemented");
  }

  void FillInts(Tensor* tensor) const override;

  void FillFloats(Tensor* tensor) const override;

  void FillStrings(Tensor* tensor) const override;

private:
	int row_index_;
  const std::vector<int>& i32_indexes_;
  const std::vector<int>& i64_indexes_;
  const std::vector<int>& f32_indexes_;
  const std::vector<int>& f64_indexes_;
  const std::vector<int>& s_indexes_;
  const std::vector<int>& ls_indexes_;
  const std::vector<const void*>& table_accessors_;
};

} // namespace io
} // namespace graphlearn

#endif

#endif // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_STORAGE_UTILS_H_
