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

#include "actor/tensor_serializer.h"

#include "seastar/core/deleter.hh"
#include "service/tensor_impl.h"

namespace graphlearn {
namespace act {

void TensorSerializer::dump_to(hiactor::serializable_queue& qu) {
  auto data_type = impl_->DType();
  auto data_size = impl_->Size();

  // Write type and size.
  static const uint32_t meta_size = sizeof(DataType) + sizeof(int32_t);
  auto meta_buf = seastar::temporary_buffer<char>(meta_size);
  memcpy(meta_buf.get_write(), &data_type, sizeof(DataType));
  memcpy(meta_buf.get_write() + sizeof(DataType), &data_size, sizeof(int32_t));
  qu.push(std::move(meta_buf));

  // Write contents with zero-copy
  char* ptr = nullptr;
  size_t len = 0;
  if (data_type == DataType::kString) {
    for (int i = 0; i < data_size; ++i) {
      auto &str = GetString(i);
      len = str.size();
      ptr = const_cast<char*>(str.data());
      std::shared_ptr<TensorImpl> impl_share = impl_;
      auto deleter = seastar::make_object_deleter(std::move(impl_share));
      qu.push(seastar::temporary_buffer<char>(ptr, len, std::move(deleter)));
    }
    return;
  }
  if (data_type == DataType::kInt32) {
    len = sizeof(int32_t) * data_size;
    ptr = const_cast<char*>(reinterpret_cast<const char*>(GetInt32()));
  } else if (data_type == DataType::kInt64) {
    len = sizeof(int64_t) * data_size;
    ptr = const_cast<char*>(reinterpret_cast<const char*>(GetInt64()));
  } else if (data_type == DataType::kFloat) {
    len = sizeof(float) * data_size;
    ptr = const_cast<char*>(reinterpret_cast<const char*>(GetFloat()));
  } else if (data_type == DataType::kDouble) {
    len = sizeof(double) * data_size;
    ptr = const_cast<char*>(reinterpret_cast<const char*>(GetDouble()));
  }
  std::shared_ptr<TensorImpl> impl_share = impl_;
  auto deleter = seastar::make_object_deleter(std::move(impl_share));
  qu.push(seastar::temporary_buffer<char>(ptr, len, std::move(deleter)));
}

Tensor TensorSerializer::load_from(hiactor::serializable_queue &qu) {
  auto meta_buf = qu.pop();
  char *meta_ptr = meta_buf.get_write();
  auto data_type = *reinterpret_cast<DataType*>(meta_ptr);
  auto data_size = *reinterpret_cast<int32_t*>(meta_ptr + sizeof(DataType));

  Tensor tensor(data_type, data_size);
  if (data_type == DataType::kInt32) {
    auto data_buf = qu.pop();
    auto int32_ptr = reinterpret_cast<const int32_t*>(data_buf.get());
    for (int i = 0; i < data_size; ++i) {
      tensor.AddInt32(int32_ptr[i]);
    }
  } else if (data_type == DataType::kInt64) {
    auto data_buf = qu.pop();
    auto int64_ptr = reinterpret_cast<const int64_t*>(data_buf.get());
    for (int i = 0; i < data_size; ++i) {
      tensor.AddInt64(int64_ptr[i]);
    }
  } else if (data_type == DataType::kFloat) {
    auto data_buf = qu.pop();
    auto float_ptr = reinterpret_cast<const float*>(data_buf.get());
    for (int i = 0; i < data_size; ++i) {
      tensor.AddFloat(float_ptr[i]);
    }
  } else if (data_type == DataType::kDouble) {
    auto data_buf = qu.pop();
    auto double_ptr = reinterpret_cast<const double*>(data_buf.get());
    for (int i = 0; i < data_size; ++i) {
      tensor.AddDouble(double_ptr[i]);
    }
  } else if (data_type == DataType::kString) {
    for (int i = 0; i < data_size; ++i) {
      auto data_buf = qu.pop();
      tensor.AddString(std::string(data_buf.get(), data_buf.size()));
    }
  }
  return tensor;
}

void TensorMapSerializer::dump_to(hiactor::serializable_queue &qu) {
  for (auto& it : tensors_) {
    // write key
    // FIXME(xiaoming.qxm): zero-copy.
    auto length = it.first.size();
    auto buf = seastar::temporary_buffer<char>(length);
    memcpy(buf.get_write(), it.first.data(), length);
    qu.push(std::move(buf));

    // write value
    TensorSerializer value_serializer(it.second);
    value_serializer.dump_to(qu);
  }

  for (auto& it : sparse_tensors_) {
    auto length = it.first.size();
    auto buf = seastar::temporary_buffer<char>(length);
    memcpy(buf.get_write(), it.first.data(), length);
    qu.push(std::move(buf));

    // write valus
    TensorSerializer value_serializer(it.second.Values());
    value_serializer.dump_to(qu);

    // write segments
    TensorSerializer segment_serializer(it.second.Segments());
    segment_serializer.dump_to(qu);

  }
}

TensorMapSerializer TensorMapSerializer::load_from(hiactor::serializable_queue &qu) {
  TensorMapSerializer tm;
  while (!qu.empty()) {
    // get key
    auto buf = qu.pop();
    // FIXME(xiaoming.qxm): zero-copy.
    char *key_ptr = buf.get_write();
    auto key = std::string(buf.get_write(), buf.size());
    // get value
    auto value = TensorSerializer::load_from(qu);
    // try get segment
    auto segment = TensorSerializer::load_from(qu);
    tm.Add(key, std::move(segment), std::move(value));
  }
  return tm;
}

}  // namespace act
}  // namespace graphlearn