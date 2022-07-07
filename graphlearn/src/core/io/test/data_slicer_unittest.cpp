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

#include "core/io/data_slicer.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::io;  // NOLINT [build/namespaces]

class DataSlicerTest : public ::testing::Test {
protected:
  void SetUp() override {
  }

  void TearDown() override {
  }
};

TEST_F(DataSlicerTest, LocalSlice) {
  DataSlicer slicer(0, 1, 10);
  EXPECT_EQ(slicer.Size(), 10);
  EXPECT_EQ(slicer.LocalSize(), 10);
  EXPECT_EQ(slicer.LocalStart(), 0);
}

TEST_F(DataSlicerTest, DivisibleSlice) {
  DataSlicer slicer(3, 5, 10);
  EXPECT_EQ(slicer.Size(), 10);
  EXPECT_EQ(slicer.LocalSize(), 2);
  EXPECT_EQ(slicer.LocalStart(), 6);
}

TEST_F(DataSlicerTest, UndivisibleSlice) {
  DataSlicer slicer(0, 5, 11);
  EXPECT_EQ(slicer.Size(), 11);
  EXPECT_EQ(slicer.LocalSize(), 3);
  EXPECT_EQ(slicer.LocalStart(), 0);

  DataSlicer slicer2(1, 5, 11);
  EXPECT_EQ(slicer2.Size(), 11);
  EXPECT_EQ(slicer2.LocalSize(), 2);
  EXPECT_EQ(slicer2.LocalStart(), 3);

  DataSlicer slicer3(4, 5, 11);
  EXPECT_EQ(slicer3.Size(), 11);
  EXPECT_EQ(slicer3.LocalSize(), 2);
  EXPECT_EQ(slicer3.LocalStart(), 9);
}
