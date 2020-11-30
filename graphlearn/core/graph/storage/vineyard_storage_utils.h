#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_STORAGE_UTILS_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_STORAGE_UTILS_H_

#include <memory>

#if defined(WITH_VINEYARD)
#include "vineyard/graph/fragment/arrow_fragment.h"
#endif

#if defined(WITH_VINEYARD)
#include "graphlearn/core/graph/storage/edge_storage.h"
#include "graphlearn/core/graph/storage/graph_storage.h"
#include "graphlearn/core/graph/storage/node_storage.h"
#include "graphlearn/core/graph/storage/topo_storage.h"
#include "graphlearn/include/config.h"

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
using graphlearn::io::IndexList;
using graphlearn::io::IndexType;
using graphlearn::io::NewDataHeldAttributeValue;
using graphlearn::io::SideInfo;

void init_table_accessors(std::shared_ptr<arrow::Table> const &table,
                          size_t start_index, size_t end_index,
                          std::vector<int> &i32_indexes,
                          std::vector<int> &i64_indexes,
                          std::vector<int> &f32_indexes,
                          std::vector<int> &f64_indexes,
                          std::vector<int> &s_indexes,
                          std::vector<int> &ls_indexes,
                          std::vector<const void*> &table_accessors);

AttributeValue *
arrow_line_to_attribute_value(std::shared_ptr<arrow::Table> table,
                              int row_index, int start_index, int end_index);

AttributeValue *arrow_line_to_attribute_value_fast(
                          const int row_index,
                          const std::vector<int> &i32_indexes,
                          const std::vector<int> &i64_indexes,
                          const std::vector<int> &f32_indexes,
                          const std::vector<int> &f64_indexes,
                          const std::vector<int> &s_indexes,
                          const std::vector<int> &ls_indexes,
                          const std::vector<const void*> &table_accessors);

const IdArray get_all_src_ids(std::shared_ptr<gl_frag_t> const &frag,
                              const label_id_t edge_label);

const IdArray get_all_dst_ids(std::shared_ptr<gl_frag_t> const &frag,
                              const label_id_t edge_label);

const IndexList *get_all_in_degree(std::shared_ptr<gl_frag_t> const &frag,
                                   const label_id_t edge_label);

const IndexList *get_all_out_degree(std::shared_ptr<gl_frag_t> const &frag,
                                    const label_id_t edge_label);

const Array<IdType>
get_all_outgoing_neighbor_nodes(std::shared_ptr<gl_frag_t> const &frag,
                                IdType src_id, const label_id_t edge_label);

const Array<IdType>
get_all_outgoing_neighbor_edges(std::shared_ptr<gl_frag_t> const &frag,
                                IdType src_id, const label_id_t edge_label);

IdType get_edge_src_id(std::shared_ptr<gl_frag_t> const &frag,
                       label_id_t const edge_label,
                       std::vector<IdType> const &src_ids, IdType edge_id);

IdType get_edge_dst_id(std::shared_ptr<gl_frag_t> const &frag,
                       label_id_t const edge_label,
                       std::vector<IdType> const &dst_ids, IdType edge_id);

float get_edge_weight(std::shared_ptr<gl_frag_t> const &frag,
                      label_id_t const edge_label, IdType edge_id);

int32_t get_edge_label(std::shared_ptr<gl_frag_t> const &frag,
                       label_id_t const edge_label, IdType edge_id);

Attribute get_edge_attribute(std::shared_ptr<gl_frag_t> const &frag,
                             label_id_t const edge_label, IdType edge_id);

void init_src_dst_list(std::shared_ptr<gl_frag_t> const &frag,
                    label_id_t const edge_label, std::vector<IdType> &src_lists,
                    std::vector<IdType> &dst_lists);

SideInfo *frag_edge_side_info(std::shared_ptr<gl_frag_t> const &frag,
                              label_id_t const edge_label);

SideInfo *frag_node_side_info(std::shared_ptr<gl_frag_t> const &frag,
                              label_id_t const node_label);

int64_t find_index_of_name(std::shared_ptr<arrow::Schema> const &schema,
                           std::string const &name);

class ArrowRefAttributeValue : public AttributeValue {
public:
  ArrowRefAttributeValue(const int row_index,
                          const std::vector<int> &i32_indexes,
                          const std::vector<int> &i64_indexes,
                          const std::vector<int> &f32_indexes,
                          const std::vector<int> &f64_indexes,
                          const std::vector<int> &s_indexes,
                          const std::vector<int> &ls_indexes,
                          const std::vector<const void*> &table_accessors):
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
	const int row_index_;
  const std::vector<int> &i32_indexes_;
  const std::vector<int> &i64_indexes_;
  const std::vector<int> &f32_indexes_;
  const std::vector<int> &f64_indexes_;
  const std::vector<int> &s_indexes_;
  const std::vector<int> &ls_indexes_;
  const std::vector<const void*> &table_accessors_;
};

} // namespace io
} // namespace graphlearn

#endif

#endif // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_STORAGE_UTILS_H_
