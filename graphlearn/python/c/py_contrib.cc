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

#include <cstdint>
#include <typeinfo>

#include "contrib/knn/knn_request.h"
#include "python/c/py_bind.h"

using namespace graphlearn;

namespace {

void ImportNumpy() {
  import_array1();
}

}  // anonymous namespace

void init_contrib_module(py::module& m) {
  // KNN
  DEF_REQ(KnnRequest);
  DEF_RES(KnnResponse);

  m.def("new_knn_request",
        [](const std::string& node_type, int32_t k) {
          KnnRequest* req = new KnnRequest(node_type, k);
          return static_cast<OpRequest*>(req);
        },
        py::return_value_policy::reference,
        py::arg("node_type"),
        py::arg("k"));

  m.def("set_knn_request",
        [](OpRequest* req,
           int32_t batch_size,
           int32_t dimension,
           py::object inputs) {
          ImportNumpy();
          PyArrayObject* input = reinterpret_cast<PyArrayObject*>(inputs.ptr());
          KnnRequest* knn_req = static_cast<KnnRequest*>(req);
          knn_req->Set(reinterpret_cast<float*>(PyArray_DATA(input)),
            batch_size, dimension);
        });

  m.def("new_knn_response",
        []() {
          return static_cast<OpResponse*>(new KnnResponse());
        },
        py::return_value_policy::reference);

  m.def("get_knn_ids",
        [](OpResponse* res) {
          ImportNumpy();
          KnnResponse* knn_res = static_cast<KnnResponse*>(res);
          npy_intp shape[1];
          shape[0] = knn_res->BatchSize() * knn_res->K();
          PyArray_Descr* descr = PyArray_DescrFromType(NPY_INT64);
          PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
          PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
          memcpy(PyArray_DATA(np_array), knn_res->Ids(),
                 shape[0] * INT64_BYTES);
          CAST_RETURN(obj);
        },
        py::return_value_policy::reference);

  m.def("get_knn_distances",
        [](OpResponse* res) {
          ImportNumpy();
          KnnResponse* knn_res = static_cast<KnnResponse*>(res);
          npy_intp shape[1];
          shape[0] = knn_res->BatchSize() * knn_res->K();
          PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT32);
          PyObject* obj = PyArray_Zeros(1, shape, descr, 0);
          PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
          memcpy(PyArray_DATA(np_array), knn_res->Distances(),
                 shape[0] * FLOAT32_BYTES);
          CAST_RETURN(obj);
        },
        py::return_value_policy::reference);
}
