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

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <unordered_map>
#include <string>
#include <vector>

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#include "graphlearn/common/base/log.h"
#include "graphlearn/common/threading/sync/cond.h"
#include "graphlearn/include/sampling_request.h"
#include "graphlearn/include/aggregating_request.h"
#include "graphlearn/platform/env.h"

namespace py = pybind11;


#define INT64_BYTES sizeof(int64_t)
#define INT32_BYTES sizeof(int32_t)
#define MAX_SHARD 10
#define COUNT_PER_SHARD 50000
#define DOUBLE_BYTES sizeof(double)
#define FLOAT32_BYTES sizeof(float)

// Required to user PyArray_* functions.
void ImportNumpy() {
  import_array1();
}

const std::unordered_map<int, ::graphlearn::DataType> type_map{
  {NPY_INT32, ::graphlearn::kInt32},
  {NPY_INT64, ::graphlearn::kInt64},
  {NPY_FLOAT, ::graphlearn::kFloat},
  {NPY_DOUBLE, ::graphlearn::kDouble}};

// Here we implement new and delete req and res and info.
::graphlearn::SamplingRequest* new_nbr_req(
  const std::string& type,
  const std::string& strategy,
  int32_t neighbor_count
  ) {
  return new ::graphlearn::SamplingRequest(
    type, strategy, neighbor_count);
}

void del_nbr_req(::graphlearn::SamplingRequest* req) {
  delete req;
  req = NULL;
}

::graphlearn::SamplingResponse* new_nbr_res() {
  return new ::graphlearn::SamplingResponse();
}

void del_nbr_res(::graphlearn::SamplingResponse* res) {
  delete res;
  res = NULL;
}

void set_nbr_req(::graphlearn::SamplingRequest* req,
                 PyObject* src_ids) {
  PyArrayObject* srcs = reinterpret_cast<PyArrayObject*>(src_ids);
  npy_intp batch_size = PyArray_Size(src_ids);
  req->Set(reinterpret_cast<int64_t*>(PyArray_DATA(srcs)),
           batch_size);
}


// get output tensors_ using specified name,
// sampler does not support string output tensor.
PyObject* get_nbr_res_nbr_ids(::graphlearn::SamplingResponse* res) {
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

PyObject* get_nbr_res_edge_ids(::graphlearn::SamplingResponse* res) {
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

PyObject* get_nbr_res_degrees(::graphlearn::SamplingResponse* res) {
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

::graphlearn::GetEdgesRequest* new_get_edge_req(const std::string& edge_type,
                                                const std::string& strategy,
                                                int32_t batch_size,
                                                int32_t epoch) {
  return new ::graphlearn::GetEdgesRequest(edge_type, strategy, batch_size, epoch);
}

void del_get_edge_req(::graphlearn::GetEdgesRequest* req) {
  delete req;
  req = NULL;
}

::graphlearn::GetEdgesResponse* new_get_edge_res() {
  return new ::graphlearn::GetEdgesResponse();
}

void del_get_edge_res(::graphlearn::GetEdgesResponse* res) {
  delete res;
  res = NULL;
}

PyObject* get_edge_src_id_res(::graphlearn::GetEdgesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT64);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->SrcIds(),
         res->Size() * INT64_BYTES);
  return obj;
}

PyObject* get_edge_dst_id_res(::graphlearn::GetEdgesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT64);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->DstIds(),
         res->Size() * INT64_BYTES);
  return obj;
}

PyObject* get_edge_edge_id_res(::graphlearn::GetEdgesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT64);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->EdgeIds(),
         res->Size() * INT64_BYTES);
  return obj;
}

::graphlearn::LookupEdgesRequest* new_lookup_edges_req(const std::string& edge_type) {
  return new ::graphlearn::LookupEdgesRequest(edge_type);
}

void del_lookup_edges_req(::graphlearn::LookupEdgesRequest* req) {
  delete req;
  req = NULL;
}

// TODO: src_ids and edge_ids mast be same size
void set_lookup_edges_req(::graphlearn::LookupEdgesRequest* req,
                         PyObject* src_ids,
                         PyObject* edge_ids) {
  PyArrayObject* srcs = reinterpret_cast<PyArrayObject*>(src_ids);
  PyArrayObject* edges = reinterpret_cast<PyArrayObject*>(edge_ids);
  npy_intp batch_size = PyArray_Size(src_ids);
  req->Set(reinterpret_cast<int64_t*>(PyArray_DATA(edges)),
           reinterpret_cast<int64_t*>(PyArray_DATA(srcs)),
           batch_size);
}

::graphlearn::LookupEdgesResponse* new_lookup_edges_res() {
  return new ::graphlearn::LookupEdgesResponse();
}

void del_lookup_edges_res(::graphlearn::LookupEdgesResponse* res) {
  delete res;
  res = NULL;
}

PyObject* get_int_attr_res(::graphlearn::LookupResponse* res) {
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

PyObject* get_float_attr_res(::graphlearn::LookupResponse* res) {
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

PyObject* get_string_attr_res(::graphlearn::LookupResponse* res) {
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

PyObject* get_edge_int_attr_res(
  ::graphlearn::LookupEdgesResponse* res) {
  return get_int_attr_res(res);
}

PyObject* get_edge_float_attr_res(
  ::graphlearn::LookupEdgesResponse* res) {
  return get_float_attr_res(res);
}

PyObject* get_edge_string_attr_res(
  ::graphlearn::LookupEdgesResponse* res) {
  return get_string_attr_res(res);
}

PyObject* get_edge_weights_res(::graphlearn::LookupEdgesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT32);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->Weights(),
         res->Size() * FLOAT32_BYTES);
  return obj;
}

PyObject* get_edge_labels_res(::graphlearn::LookupEdgesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT32);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->Labels(),
         res->Size() * INT32_BYTES);
  return obj;
}

::graphlearn::GetNodesRequest* new_get_node_req(const std::string& type,
                                                const std::string& strategy,
                                                ::graphlearn::NodeFrom node_from,
                                                int32_t batch_size,
                                                int32_t epoch) {
  return new ::graphlearn::GetNodesRequest(
    type, strategy, node_from, batch_size, epoch);
}

void del_get_node_req(::graphlearn::GetNodesRequest* req) {
  delete req;
  req = NULL;
}

::graphlearn::GetNodesResponse* new_get_node_res() {
  return new ::graphlearn::GetNodesResponse();
}

void del_get_node_res(::graphlearn::GetNodesResponse* res) {
  delete res;
  res = NULL;
}

PyObject* get_node_node_id_res(::graphlearn::GetNodesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT64);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->NodeIds(),
         res->Size() * INT64_BYTES);
  return obj;
}

::graphlearn::LookupNodesRequest* new_lookup_nodes_req(
    const std::string& node_type) {
  return new ::graphlearn::LookupNodesRequest(node_type);
}

void del_lookup_nodes_req(::graphlearn::LookupNodesRequest* req) {
  delete req;
  req = NULL;
}

void set_lookup_nodes_req(::graphlearn::LookupNodesRequest* req,
                          PyObject* node_ids) {
  PyArrayObject* nodes = reinterpret_cast<PyArrayObject*>(node_ids);
  npy_intp batch_size = PyArray_Size(node_ids);
  req->Set(reinterpret_cast<int64_t*>(PyArray_DATA(nodes)),
           batch_size);
}

::graphlearn::LookupNodesResponse* new_lookup_nodes_res() {
  return new ::graphlearn::LookupNodesResponse();
}

void del_lookup_nodes_res(::graphlearn::LookupNodesResponse* res) {
  delete res;
  res = NULL;
}

PyObject* get_node_weights_res(::graphlearn::LookupNodesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT32);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->Weights(),
         res->Size() * FLOAT32_BYTES);
  return obj;
}

PyObject* get_node_labels_res(::graphlearn::LookupNodesResponse* res) {
  npy_intp shape[1];
  shape[0] = res->Size();
  PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT32);
  PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  memcpy(PyArray_DATA(np_array), res->Labels(),
         res->Size() * INT32_BYTES);
  return obj;
}

PyObject* get_node_int_attr_res(
  ::graphlearn::LookupNodesResponse* res) {
  return get_int_attr_res(res);
}

PyObject* get_node_float_attr_res(
  ::graphlearn::LookupNodesResponse* res) {
  return get_float_attr_res(res);
}

PyObject* get_node_string_attr_res(
  ::graphlearn::LookupNodesResponse* res) {
  return get_string_attr_res(res);
}

::graphlearn::AggregatingRequest* new_agg_nodes_req(
    const std::string& node_type,
    const std::string& strategy) {
  return new ::graphlearn::AggregatingRequest(node_type, strategy);
}

void del_agg_nodes_req(::graphlearn::AggregatingRequest* req) {
  delete req;
  req = NULL;
}

void set_agg_nodes_req(::graphlearn::AggregatingRequest* req,
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

::graphlearn::AggregatingResponse* new_agg_nodes_res() {
  return new ::graphlearn::AggregatingResponse();
}

void del_agg_nodes_res(::graphlearn::AggregatingResponse* res) {
  delete res;
  res = NULL;
}

PyObject* get_node_agg_res(::graphlearn::AggregatingResponse* res) {
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

#endif // GRAPHLEARN_PYTHON_PY_WRAPPER_H_
