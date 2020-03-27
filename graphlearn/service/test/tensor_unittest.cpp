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

#include <string>
#include "graphlearn/include/tensor.h"
#include "google/protobuf/repeated_field.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

TEST(TensorTest, Int32) {
  Tensor t(kInt32, 10);

  // Size
  EXPECT_EQ(t.DType(), kInt32);
  EXPECT_EQ(t.Size(), 0);
  // Add
  for (int32_t i = 0; i < 10; ++i) {
    t.AddInt32(i);
    EXPECT_EQ(t.Size(), i + 1);
  }
  // Get
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(t.GetInt32(i), i);
  }
  // Get all
  const int32_t* buf = t.GetInt32();
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(buf[i], i);
  }

  // Swap
  Tensor t2(kInt32);
  t2.Swap(t);

  EXPECT_EQ(t.Size(), 0);
  EXPECT_EQ(t2.Size(), 10);

  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(t2.GetInt32(i), i);
  }

  const int32_t* buf2 = t2.GetInt32();
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(buf2[i], i);
  }

  // Copy from PB
  ::google::protobuf::RepeatedField<int32_t> tmp_buf;
  for (int32_t i = 0; i < 8; ++i) {
    tmp_buf.Add(i);
  }
  EXPECT_EQ(tmp_buf.size(), 8);

  Tensor t3(kInt32);
  t3.CopyFromPB(&tmp_buf);

  EXPECT_EQ(t3.Size(), 8);
  EXPECT_EQ(tmp_buf.size(), 8);

  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(t3.GetInt32(i), i);
  }

  const int32_t* buf3 = t3.GetInt32();
  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(buf3[i], i);
  }

  // Swap from PB
  EXPECT_EQ(tmp_buf.size(), 8);

  Tensor t4(kInt32);
  t4.SwapWithPB(&tmp_buf);

  EXPECT_EQ(t4.Size(), 8);
  EXPECT_EQ(tmp_buf.size(), 0);

  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(t4.GetInt32(i), i);
  }

  const int32_t* buf4 = t4.GetInt32();
  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(buf4[i], i);
  }
}

TEST(TensorTest, Int64) {
  Tensor t(kInt64, 10);

  // Size
  EXPECT_EQ(t.DType(), kInt64);
  EXPECT_EQ(t.Size(), 0);
  // Add
  for (int64_t i = 0; i < 10; ++i) {
    t.AddInt64(i);
    EXPECT_EQ(t.Size(), i + 1);
  }
  // Get
  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_EQ(t.GetInt64(i), i);
  }
  // Get all
  const int64_t* buf = t.GetInt64();
  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_EQ(buf[i], i);
  }

  // Swap
  Tensor t2(kInt64);
  t2.Swap(t);

  EXPECT_EQ(t.Size(), 0);
  EXPECT_EQ(t2.Size(), 10);

  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_EQ(t2.GetInt64(i), i);
  }

  const int64_t* buf2 = t2.GetInt64();
  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_EQ(buf2[i], i);
  }

  // Copy from PB
  ::google::protobuf::RepeatedField<int64_t> tmp_buf;
  for (int64_t i = 0; i < 8; ++i) {
    tmp_buf.Add(i);
  }
  EXPECT_EQ(tmp_buf.size(), 8);

  Tensor t3(kInt64);
  t3.CopyFromPB(&tmp_buf);

  EXPECT_EQ(t3.Size(), 8);
  EXPECT_EQ(tmp_buf.size(), 8);

  for (int64_t i = 0; i < 8; ++i) {
    EXPECT_EQ(t3.GetInt64(i), i);
  }

  const int64_t* buf3 = t3.GetInt64();
  for (int64_t i = 0; i < 8; ++i) {
    EXPECT_EQ(buf3[i], i);
  }

  // Swap from PB
  EXPECT_EQ(tmp_buf.size(), 8);

  Tensor t4(kInt64);
  t4.SwapWithPB(&tmp_buf);

  EXPECT_EQ(t4.Size(), 8);
  EXPECT_EQ(tmp_buf.size(), 0);

  for (int64_t i = 0; i < 8; ++i) {
    EXPECT_EQ(t4.GetInt64(i), i);
  }

  const int64_t* buf4 = t4.GetInt64();
  for (int64_t i = 0; i < 8; ++i) {
    EXPECT_EQ(buf4[i], i);
  }
}

TEST(TensorTest, Float) {
  Tensor t(kFloat, 10);

  // Size
  EXPECT_EQ(t.DType(), kFloat);
  EXPECT_EQ(t.Size(), 0);
  // Add
  for (int32_t i = 0; i < 10; ++i) {
    t.AddFloat(float(i));
    EXPECT_EQ(t.Size(), i + 1);
  }
  // Get
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(t.GetFloat(i), float(i));
  }
  // Get all
  const float* buf = t.GetFloat();
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(buf[i], float(i));
  }

  // Swap
  Tensor t2(kFloat);
  t2.Swap(t);

  EXPECT_EQ(t.Size(), 0);
  EXPECT_EQ(t2.Size(), 10);

  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(t2.GetFloat(i), float(i));
  }

  const float* buf2 = t2.GetFloat();
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(buf2[i], float(i));
  }

  // Copy from PB
  ::google::protobuf::RepeatedField<float> tmp_buf;
  for (int32_t i = 0; i < 8; ++i) {
    tmp_buf.Add(float(i));
  }
  EXPECT_EQ(tmp_buf.size(), 8);

  Tensor t3(kFloat);
  t3.CopyFromPB(&tmp_buf);

  EXPECT_EQ(t3.Size(), 8);
  EXPECT_EQ(tmp_buf.size(), 8);

  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(t3.GetFloat(i), float(i));
  }

  const float* buf3 = t3.GetFloat();
  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(buf3[i], float(i));
  }

  // Swap from PB
  EXPECT_EQ(tmp_buf.size(), 8);

  Tensor t4(kFloat);
  t4.SwapWithPB(&tmp_buf);

  EXPECT_EQ(t4.Size(), 8);
  EXPECT_EQ(tmp_buf.size(), 0);

  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(t4.GetFloat(i), float(i));
  }

  const float* buf4 = t4.GetFloat();
  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(buf4[i], float(i));
  }
}

TEST(TensorTest, Double) {
  Tensor t(kDouble, 10);

  // Size
  EXPECT_EQ(t.DType(), kDouble);
  EXPECT_EQ(t.Size(), 0);
  // Add
  for (int32_t i = 0; i < 10; ++i) {
    t.AddDouble(double(i));
    EXPECT_EQ(t.Size(), i + 1);
  }
  // Get
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_DOUBLE_EQ(t.GetDouble(i), double(i));
  }
  // Get all
  const double* buf = t.GetDouble();
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_DOUBLE_EQ(buf[i], double(i));
  }

  // Swap
  Tensor t2(kDouble);
  t2.Swap(t);

  EXPECT_EQ(t.Size(), 0);
  EXPECT_EQ(t2.Size(), 10);

  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_DOUBLE_EQ(t2.GetDouble(i), double(i));
  }

  const double* buf2 = t2.GetDouble();
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_DOUBLE_EQ(buf2[i], double(i));
  }

  // Copy from PB
  ::google::protobuf::RepeatedField<double> tmp_buf;
  for (int32_t i = 0; i < 8; ++i) {
    tmp_buf.Add(double(i));
  }
  EXPECT_EQ(tmp_buf.size(), 8);

  Tensor t3(kDouble);
  t3.CopyFromPB(&tmp_buf);

  EXPECT_EQ(t3.Size(), 8);
  EXPECT_EQ(tmp_buf.size(), 8);

  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_DOUBLE_EQ(t3.GetDouble(i), double(i));
  }

  const double* buf3 = t3.GetDouble();
  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_DOUBLE_EQ(buf3[i], double(i));
  }

  // Swap from PB
  EXPECT_EQ(tmp_buf.size(), 8);

  Tensor t4(kDouble);
  t4.SwapWithPB(&tmp_buf);

  EXPECT_EQ(t4.Size(), 8);
  EXPECT_EQ(tmp_buf.size(), 0);

  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_DOUBLE_EQ(t4.GetDouble(i), double(i));
  }

  const double* buf4 = t4.GetDouble();
  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_DOUBLE_EQ(buf4[i], double(i));
  }
}

TEST(TensorTest, String) {
  Tensor t(kString, 10);

  // Size
  EXPECT_EQ(t.DType(), kString);
  EXPECT_EQ(t.Size(), 0);
  // Add
  for (int32_t i = 0; i < 10; ++i) {
    t.AddString(std::to_string(i));
    EXPECT_EQ(t.Size(), i + 1);
  }
  // Get
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(t.GetString(i), std::to_string(i));
  }
  // Get all
  const std::string* buf = t.GetString();
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(buf[i], std::to_string(i));
  }

  // Swap
  Tensor t2(kString);
  t2.Swap(t);

  EXPECT_EQ(t.Size(), 0);
  EXPECT_EQ(t2.Size(), 10);

  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(t2.GetString(i), std::to_string(i));
  }

  const std::string* buf2 = t2.GetString();
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(buf2[i], std::to_string(i));
  }

  // Copy from PB
  ::google::protobuf::RepeatedField<std::string> tmp_buf;
  for (int32_t i = 0; i < 8; ++i) {
    tmp_buf.Add(std::to_string(i));
  }
  EXPECT_EQ(tmp_buf.size(), 8);

  Tensor t3(kString);
  t3.CopyFromPB(&tmp_buf);

  EXPECT_EQ(t3.Size(), 8);
  EXPECT_EQ(tmp_buf.size(), 8);

  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(t3.GetString(i), std::to_string(i));
  }

  const std::string* buf3 = t3.GetString();
  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(buf3[i], std::to_string(i));
  }

  // Swap from PB
  EXPECT_EQ(tmp_buf.size(), 8);

  Tensor t4(kString);
  t4.SwapWithPB(&tmp_buf);

  EXPECT_EQ(t4.Size(), 8);
  EXPECT_EQ(tmp_buf.size(), 0);

  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(t4.GetString(i), std::to_string(i));
  }

  const std::string* buf4 = t4.GetString();
  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(buf4[i], std::to_string(i));
  }
}
