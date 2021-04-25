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

#ifndef GRAPHLEARN_PYTHON_C_PY_BIND_H_
#define GRAPHLEARN_PYTHON_C_PY_BIND_H_

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

namespace py = pybind11;

#define INT32_BYTES sizeof(int32_t)
#define INT64_BYTES sizeof(int64_t)
#define FLOAT32_BYTES sizeof(float)

#define DEF_REQ(type) py::class_<type, OpRequest>(m, #type).def(py::init<>())
#define DEF_RES(type) py::class_<type, OpResponse>(m, #type).def(py::init<>())

#define CAST_RETURN(obj) return py::reinterpret_steal<py::object>(obj)

#define CALL_FUNC(type) \
  [](Client& self, type##Request* req, type##Response* res) { \
    return self.type(req, res); \
  }

#endif  // GRAPHLEARN_PYTHON_C_PY_BIND_H_
