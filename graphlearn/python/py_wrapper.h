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

#ifndef GRAPHLEARN_PYTHON_PY_WRAPPER_H_
#define GRAPHLEARN_PYTHON_PY_WRAPPER_H_

#include <cstdarg>
#include <cstring>
#include <string>

#include "graphlearn/include/aggregating_request.h"
#include "graphlearn/include/graph_request.h"
#include "graphlearn/include/sampling_request.h"
#include "graphlearn/python/py_bind.h"

using namespace graphlearn;  // NOLINT

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
  const std::string* data = res->StringAttrs();
  for (int32_t idx = 0; idx < batch_size * attr_num; ++idx) {
    out[idx] = PyBytes_FromString(data[idx].c_str());
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
    int32_t neighbor_count) {
  return new SamplingRequest(type, strategy, neighbor_count);
}

SamplingResponse* new_sampling_response() {
  return new SamplingResponse();
}

void set_sampling_request(SamplingRequest* req, PyObject* src_ids) {
  PyArrayObject* srcs = reinterpret_cast<PyArrayObject*>(src_ids);
  npy_intp batch_size = PyArray_Size(src_ids);
  req->Set(reinterpret_cast<int64_t*>(PyArray_DATA(srcs)), batch_size);
}

PyObject* get_sampling_node_ids(SamplingResponse* res) {
  int32_t size = res->TotalNeighborCount();
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
  int32_t size = res->TotalNeighborCount();
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
  int32_t size = res->BatchSize();
  npy_intp shape[1];
  shape[0] = res->BatchSize();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT32);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(reinterpret_cast<int32_t*>(PyArray_DATA(np_array)),
         res->GetDegrees(), size * INT32_BYTES);
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

#endif  // GRAPHLEARN_PYTHON_PY_WRAPPER_H_
