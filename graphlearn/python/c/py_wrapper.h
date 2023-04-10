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

#ifndef GRAPHLEARN_PYTHON_C_PY_WRAPPER_H_
#define GRAPHLEARN_PYTHON_C_PY_WRAPPER_H_

#include <cstdarg>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "include/aggregating_request.h"
#include "include/graph_request.h"
#include "include/sampling_request.h"
#include "include/subgraph_request.h"
#include "include/sampling_request.h"
#include "generated/proto/dag.pb.h"
#include "generated/proto/request.pb.h"
#include "python/c/py_bind.h"

using namespace graphlearn;  // NOLINT

void add_dag_node_string_params(DagNodeDef* node,
                                const std::string& name,
                                const std::string& value) {
  TensorValue* param = node->add_params();
  param->set_name(name);
  param->set_dtype(kString);
  param->set_length(1);
  param->add_string_values(value);
}

void add_dag_node_int_params(DagNodeDef* node,
                             const std::string& name,
                             int32_t value) {
  TensorValue* param = node->add_params();
  param->set_name(name);
  param->set_dtype(kInt32);
  param->set_length(1);
  param->add_int32_values(value);
}

void add_dag_node_int_vector_params(DagNodeDef* node,
                                    const std::string& name,
                                    std::vector<int32_t> values) {
  TensorValue* param = node->add_params();
  param->set_name(name);
  param->set_dtype(kInt32);
  param->set_length(values.size());
  for (auto& v: values) {
    param->add_int32_values(v);
  }
}

void add_dag_node_float_vector_params(DagNodeDef* node,
                                      const std::string& name,
                                      std::vector<float> values) {
  TensorValue* param = node->add_params();
  param->set_name(name);
  param->set_dtype(kFloat);
  param->set_length(values.size());
  for (auto& v: values) {
    param->add_float_values(v);
  }
}

void add_dag_node_in_edge(DagNodeDef* node, const DagEdgeDef* edge) {
  DagEdgeDef* e = node->add_in_edges();
  *e = *edge;
}

void add_dag_node_out_edge(DagNodeDef* node, const DagEdgeDef* edge) {
  DagEdgeDef* e = node->add_out_edges();
  *e = *edge;
}

void add_dag_node(DagDef* dag, const DagNodeDef* node) {
  DagNodeDef* n = dag->add_nodes();
  *n = *node;
}

DagDef* new_dag() {
  return new DagDef();
}

std::string debug_string(DagDef* dag) {
  return dag->DebugString();
}

void set_dag_id(DagDef* dag, int32_t dag_id) {
  dag->set_id(dag_id);
}

DagEdgeDef* new_dag_edge() {
  return new DagEdgeDef();
}

void set_dag_edge_id(DagEdgeDef* edge, int32_t id) {
  edge->set_id(id);
}

void set_dag_edge_src_output(DagEdgeDef* edge,
                       const std::string& src_output) {
  edge->set_src_output(src_output);
}

void set_dag_edge_dst_input(DagEdgeDef* edge,
                       const std::string& dst_input) {
  edge->set_dst_input(dst_input);
}

DagNodeDef* new_dag_node() {
  return new DagNodeDef();
}

void set_dag_node_id(DagNodeDef* node, int32_t node_id) {
  node->set_id(node_id);
}

void set_dag_node_op_name(DagNodeDef* node, const std::string& op_name) {
  node->set_op_name(op_name);
}

GetNodesRequest* new_get_nodes_request(
    const std::string& type,
    const std::string& strategy,
    NodeFrom node_from,
    int32_t batch_size,
    int32_t epoch) {
  return new GetNodesRequest(
    type, strategy, node_from, batch_size, epoch);
}

GetNodesResponse* new_get_nodes_response() {
  return new GetNodesResponse();
}

PyObject* get_node_ids(GetNodesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT64);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->NodeIds(),
         res->Size() * INT64_BYTES);
  return obj;
}

LookupNodesRequest* new_lookup_nodes_request(
    const std::string& node_type) {
  return new LookupNodesRequest(node_type);
}

void set_lookup_nodes_request(
    LookupNodesRequest* req,
    PyObject* node_ids) {
  PyArrayObject* nodes = reinterpret_cast<PyArrayObject*>(node_ids);
  npy_intp batch_size = PyArray_Size(node_ids);
  req->Set(reinterpret_cast<int64_t*>(PyArray_DATA(nodes)),
           batch_size);
}

LookupNodesResponse* new_lookup_nodes_response() {
  return new LookupNodesResponse();
}

PyObject* get_node_weights(LookupNodesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT32);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->Weights(),
         res->Size() * FLOAT32_BYTES);
  return obj;
}

PyObject* get_node_labels(LookupNodesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT32);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->Labels(),
         res->Size() * INT32_BYTES);
  return obj;
}

PyObject* get_int_attributes(LookupResponse* res) {
  int32_t attr_num = res->IntAttrNum();
  if (attr_num <= 0) {
    Py_RETURN_NONE;
  }

  npy_intp shape[1];
  int32_t batch_size = res->Size();
  shape[0] = batch_size * attr_num;
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT64);
  PyObject* obj = PyArray_Empty(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->IntAttrs(),
         batch_size * attr_num * INT64_BYTES);
  return obj;
}

PyObject* get_float_attributes(LookupResponse* res) {
  int32_t attr_num = res->FloatAttrNum();
  if (attr_num <= 0) {
    Py_RETURN_NONE;
  }

  npy_intp shape[1];
  int32_t batch_size = res->Size();
  shape[0] = batch_size * attr_num;
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT32);
  PyObject* obj = PyArray_Empty(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->FloatAttrs(),
         batch_size * attr_num * FLOAT32_BYTES);
  return obj;
}

PyObject* get_string_attributes(LookupResponse* res) {
  int32_t attr_num = res->StringAttrNum();
  if (attr_num <= 0) {
    Py_RETURN_NONE;
  }

  npy_intp shape[1];
  int32_t batch_size = res->Size();
  shape[0] = batch_size * attr_num;
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_OBJECT);
  PyObject* obj = PyArray_Empty(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  PyObject** out = reinterpret_cast<PyObject**>(PyArray_DATA(np_array));
  const std::string* const* data = res->StringAttrs();
  for (int32_t idx = 0; idx < batch_size * attr_num; ++idx) {
    out[idx] = PyBytes_FromString(data[idx]->c_str());
  }
  return obj;
}

PyObject* get_node_int_attributes(LookupNodesResponse* res) {
  return get_int_attributes(res);
}

PyObject* get_node_float_attributes(LookupNodesResponse* res) {
  return get_float_attributes(res);
}

PyObject* get_node_string_attributes(LookupNodesResponse* res) {
  return get_string_attributes(res);
}

GetEdgesRequest* new_get_edges_request(
    const std::string& edge_type,
    const std::string& strategy,
    int32_t batch_size,
    int32_t epoch) {
  return new GetEdgesRequest(edge_type, strategy, batch_size, epoch);
}

GetEdgesResponse* new_get_edges_response() {
  return new GetEdgesResponse();
}

PyObject* get_edge_src_id(GetEdgesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT64);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->SrcIds(),
         res->Size() * INT64_BYTES);
  return obj;
}

PyObject* get_edge_dst_id(GetEdgesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT64);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->DstIds(),
         res->Size() * INT64_BYTES);
  return obj;
}

PyObject* get_edge_id(GetEdgesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT64);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->EdgeIds(),
         res->Size() * INT64_BYTES);
  return obj;
}

LookupEdgesRequest* new_lookup_edges_request(const std::string& edge_type) {
  return new LookupEdgesRequest(edge_type);
}

void set_lookup_edges_request(
    LookupEdgesRequest* req,
    PyObject* src_ids,
    PyObject* edge_ids) {
  PyArrayObject* srcs = reinterpret_cast<PyArrayObject*>(src_ids);
  PyArrayObject* edges = reinterpret_cast<PyArrayObject*>(edge_ids);
  npy_intp batch_size = PyArray_Size(src_ids);
  req->Set(reinterpret_cast<int64_t*>(PyArray_DATA(edges)),
           reinterpret_cast<int64_t*>(PyArray_DATA(srcs)),
           batch_size);
}

LookupEdgesResponse* new_lookup_edges_response() {
  return new LookupEdgesResponse();
}

PyObject* get_edge_weights(LookupEdgesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT32);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->Weights(),
         res->Size() * FLOAT32_BYTES);
  return obj;
}

PyObject* get_edge_labels(LookupEdgesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT32);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->Labels(),
         res->Size() * INT32_BYTES);
  return obj;
}

PyObject* get_edge_int_attributes(LookupEdgesResponse* res) {
  return get_int_attributes(res);
}

PyObject* get_edge_float_attributes(LookupEdgesResponse* res) {
  return get_float_attributes(res);
}

PyObject* get_edge_string_attributes(LookupEdgesResponse* res) {
  return get_string_attributes(res);
}

SamplingRequest* new_sampling_request(
    const std::string& type,
    const std::string& strategy,
    int32_t neighbor_count,
    FilterType filter_type,
    FilterField filter_field) {
  return new SamplingRequest(type, strategy, neighbor_count, filter_type, filter_field);
}

ConditionalSamplingRequest* new_conditional_sampling_request(
    const std::string& type,
    const std::string& strategy,
    int32_t neighbor_count,
    const std::string& dst_node_type,
    bool batch_share,
    bool unique) {
  return new ConditionalSamplingRequest(type, strategy, neighbor_count,
      dst_node_type, batch_share, unique);
}

SamplingResponse* new_sampling_response() {
  return new SamplingResponse();
}

void set_sampling_request(SamplingRequest* req, PyObject* src_ids) {
  PyArrayObject* srcs = reinterpret_cast<PyArrayObject*>(src_ids);
  npy_intp batch_size = PyArray_Size(src_ids);
  req->Set(reinterpret_cast<int64_t*>(PyArray_DATA(srcs)), batch_size);
}

void set_conditional_sampling_request_ids(ConditionalSamplingRequest* req,
                                          PyObject* src_ids,
                                          PyObject* dst_ids) {
  PyArrayObject* srcs = reinterpret_cast<PyArrayObject*>(src_ids);
  PyArrayObject* dsts = reinterpret_cast<PyArrayObject*>(dst_ids);
  npy_intp batch_size = PyArray_Size(src_ids);
  req->SetIds(reinterpret_cast<int64_t*>(PyArray_DATA(srcs)),
              reinterpret_cast<int64_t*>(PyArray_DATA(dsts)),
              batch_size);
}

void set_conditional_sampling_request_cols(
    ConditionalSamplingRequest* req,
    const std::vector<int32_t>& int_cols,
    const std::vector<float>& int_props,
    const std::vector<int32_t>& float_cols,
    const std::vector<float>& float_props,
    const std::vector<int32_t>& str_cols,
    const std::vector<float>& str_props) {
  req->SetSelectedCols(int_cols, int_props,
                       float_cols, float_props,
                       str_cols, str_props);
}

PyObject* get_sampling_node_ids(SamplingResponse* res) {
  int32_t size = res->GetShape().size;
  npy_intp shape[1];
  shape[0] = size;
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT64);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(reinterpret_cast<int64_t*>(PyArray_DATA(np_array)),
         res->GetNeighborIds(), size * INT64_BYTES);
  return obj;
}

PyObject* get_sampling_edge_ids(SamplingResponse* res) {
  int32_t size = res->GetShape().size;
  npy_intp shape[1];
  shape[0] = size;
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT64);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(reinterpret_cast<int64_t*>(PyArray_DATA(np_array)),
         res->GetEdgeIds(), size * INT64_BYTES);
  return obj;
}

PyObject* get_sampling_node_degrees(SamplingResponse* res) {
  int32_t size = res->GetShape().dim1;
  npy_intp shape[1];
  shape[0] = size;
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT32);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(reinterpret_cast<int32_t*>(PyArray_DATA(np_array)),
         res->GetShape().segments.data(), size * INT32_BYTES);
  return obj;
}

AggregatingRequest* new_aggregating_request(
    const std::string& node_type,
    const std::string& strategy) {
  return new AggregatingRequest(node_type, strategy);
}

void set_aggregating_request(
    AggregatingRequest* req,
    PyObject* node_ids,
    PyObject* segment_ids,
    int32_t num_segments) {
  PyArrayObject* nodes = reinterpret_cast<PyArrayObject*>(node_ids);
  npy_intp num_ids = PyArray_Size(node_ids);
  PyArrayObject* segs = reinterpret_cast<PyArrayObject*>(segment_ids);
  req->Set(reinterpret_cast<int64_t*>(PyArray_DATA(nodes)),
           reinterpret_cast<int32_t*>(PyArray_DATA(segs)),
           num_ids,
           num_segments);
}

AggregatingResponse* new_aggregating_response() {
  return new AggregatingResponse();
}

PyObject* get_aggregating_nodes(AggregatingResponse* res) {
  int32_t attr_num = res->EmbeddingDim();
  if (attr_num <= 0) {
    Py_RETURN_NONE;
  }
  npy_intp shape[1];
  int32_t batch_size = res->NumSegments();
  shape[0] = batch_size * attr_num;
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT32);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->Embeddings(),
         batch_size * attr_num *FLOAT32_BYTES);
  return obj;
}

SubGraphRequest* new_subgraph_request(
    const std::string& nbr_type,
    const std::vector<int32_t>& num_nbrs,
    bool need_dist) {
  return new SubGraphRequest(nbr_type, num_nbrs, need_dist);
}

SubGraphResponse* new_subgraph_response() {
  return new SubGraphResponse();
}

void set_subgraph_request(SubGraphRequest* req, PyObject* src_ids,
    PyObject* dst_ids) {
  PyArrayObject* srcs = reinterpret_cast<PyArrayObject*>(src_ids);
  npy_intp batch_size = PyArray_Size(src_ids);
  if (dst_ids != Py_None) {
    PyArrayObject* dsts = reinterpret_cast<PyArrayObject*>(dst_ids);
    req->Set(reinterpret_cast<int64_t*>(PyArray_DATA(srcs)),
             reinterpret_cast<int64_t*>(PyArray_DATA(dsts)),
             batch_size);
  } else {
    req->Set(reinterpret_cast<int64_t*>(PyArray_DATA(srcs)), batch_size);
  }
}

PyObject* get_node_set(SubGraphResponse* res) {
  npy_intp shape[1];
  shape[0] = res->NodeCount();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT64);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->NodeIds(),
         res->NodeCount() * INT64_BYTES);
  return obj;
}

PyObject* get_row_idx(SubGraphResponse* res) {
  npy_intp shape[1];
  shape[0] = res->EdgeCount();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT32);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->RowIndices(),
         res->EdgeCount() * INT32_BYTES);
  return obj;
}

PyObject* get_col_idx(SubGraphResponse* res) {
  npy_intp shape[1];
  shape[0] = res->EdgeCount();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT32);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->ColIndices(),
         res->EdgeCount() * INT32_BYTES);
  return obj;
}

PyObject* get_edge_set(SubGraphResponse* res) {
  npy_intp shape[1];
  shape[0] = res->EdgeCount();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT64);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->EdgeIds(),
         res->EdgeCount() * INT64_BYTES);
  return obj;
}

PyObject* get_dist_to_src(SubGraphResponse* res) {
  npy_intp shape[1];
  shape[0] = res->NodeCount();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT32);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->DistToSrc(),
         res->NodeCount() * INT32_BYTES);
  return obj;
}

PyObject* get_dist_to_dst(SubGraphResponse* res) {
  npy_intp shape[1];
  shape[0] = res->NodeCount();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT32);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->DistToDst(),
         res->NodeCount() * INT32_BYTES);
  return obj;
}

GetStatsRequest* new_get_stats_request() {
  return new GetStatsRequest();
}

GetStatsResponse* new_get_stats_response() {
  return new GetStatsResponse();
}

Counts get_stats(GetStatsResponse* res) {
  Counts counts;
  for(const auto& it : res->tensors_) {
    for (int32_t i = 0; i < it.second.Size(); ++i) {
      counts[it.first].push_back(it.second.GetInt32(i));
    }
  }
  return counts;
}

GetDegreeRequest* new_get_degree_request(
    const std::string& edge_type, NodeFrom node_from) {
  return new GetDegreeRequest(edge_type, node_from);
}

void set_degree_request(
    GetDegreeRequest* req, PyObject* node_ids) {
  PyArrayObject* nodes = reinterpret_cast<PyArrayObject*>(node_ids);
  npy_intp batch_size = PyArray_Size(node_ids);
  req->Set(reinterpret_cast<int64_t*>(PyArray_DATA(nodes)),
           batch_size);
}

GetDegreeResponse* new_get_degree_response() {
  return new GetDegreeResponse();
}

PyObject* get_degree(GetDegreeResponse* res) {
  npy_intp shape[1];
  int32_t batch_size = res->Size();
  shape[0] = batch_size;
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT32);
  PyObject* obj = PyArray_Empty(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->GetDegrees(),
         batch_size * INT32_BYTES);
  return obj;
}

GetDagValuesRequest* new_get_dag_values_request(
    int32_t dag_id, int32_t epoch) {
  return new GetDagValuesRequest(dag_id, epoch);
}

GetDagValuesResponse* new_get_dag_values_response() {
  return new GetDagValuesResponse();
}

PyObject* PyArrayFromFloatVector(const float* data, npy_intp* size) {
  PyObject* obj = PyArray_SimpleNewFromData(
    1, size, NPY_FLOAT32, (void *)data);
  return obj;
}

PyObject* PyArrayFromIntVector(const int32_t* data, npy_intp* size) {
  PyObject* obj = PyArray_SimpleNewFromData(
    1, size, NPY_INT32, (void *)data);
  return obj;
}

PyObject* PyArrayFromInt64Vector(const int64_t* data, npy_intp* size) {
  PyObject* obj = PyArray_SimpleNewFromData(
    1, size, NPY_INT64, (void *)data);
  return obj;
}

PyObject* PyArrayFromStringVector(
    const std::string* const* data, npy_intp* size) {
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_OBJECT);
  PyObject* obj = PyArray_Empty(1, size, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  PyObject** out = reinterpret_cast<PyObject**>(PyArray_DATA(np_array));
  for (int32_t idx = 0; idx < size[0]; ++idx) {
    out[idx] = PyBytes_FromString(data[idx]->c_str());
  }
  return obj;
}

// values
PyObject* get_dag_value(GetDagValuesResponse* res,
                        int32_t node_id,
                        const std::string& key) {
  auto t = res->GetValue(node_id, key);
  auto values = std::get<0>(t);
  if (values == nullptr) {
    return Py_None;
  }
  int32_t size = values->Size();
  npy_intp shape[1];
  shape[0] = size;

  PyObject* obj = nullptr;

  switch (values->DType()) {
  case kInt32:
    obj = PyArrayFromIntVector(values->GetInt32(), shape);
    break;
  case kInt64:
    obj = PyArrayFromInt64Vector(values->GetInt64(), shape);
    break;
  case kFloat:
    obj = PyArrayFromFloatVector(values->GetFloat(), shape);
    break;
  case kString:
    obj = PyArrayFromStringVector(values->GetString(), shape);
    break;
  default:
    break;
  }

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(obj));
}

PyObject* get_dag_value_indice(GetDagValuesResponse* res,
                              int32_t node_id,
                              const std::string& key) {

  auto t = res->GetValue(node_id, key);
  auto indices = std::get<1>(t);
  if (indices == nullptr) {
    return Py_None;
  }
  int32_t size = indices->Size();
  if (size == 0) {
    return Py_None;
  }
  npy_intp shape[1];
  shape[0] = size;

  PyObject* obj = PyArrayFromIntVector(std::get<1>(t)->GetInt32(), shape);
  return PyArray_Return(reinterpret_cast<PyArrayObject*>(obj));
}

#endif  // GRAPHLEARN_PYTHON_C_PY_WRAPPER_H_
