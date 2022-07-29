/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "gtest/gtest.h"

#include "common/log.h"
#include "common/schema.h"

using namespace dgs;

TEST(Schema, Construct) {
  InitGoogleLogging();

  auto& schema = Schema::GetInstance();
  bool ok = schema.Init("../../conf/schema.template.json",
                        "../../fbs/schema.fbs",
                        {"../../fbs/"});
  EXPECT_TRUE(ok);

  EXPECT_EQ(schema.AttrDefNum(), 5);
  auto& timestamp_def = schema.GetAttrDefByType(0);
  EXPECT_EQ(timestamp_def.Name(), "timestamp");
  EXPECT_EQ(timestamp_def.ValueType(), AttributeValueType::INT64);
  auto& weight_def = schema.GetAttrDefByName("weight");
  EXPECT_EQ(weight_def.Type(), 1);
  EXPECT_EQ(weight_def.ValueType(), AttributeValueType::FLOAT32);
  auto& attr_def_map = schema.AttrDefMap();
  EXPECT_EQ(attr_def_map.at(2).Name(), "label");
  EXPECT_EQ(attr_def_map.at(3).ValueType(), AttributeValueType::STRING);
  EXPECT_EQ(attr_def_map.at(4).ValueType(), AttributeValueType::FLOAT32_LIST);

  EXPECT_EQ(schema.VertexDefNum(), 2);
  auto& user_def = schema.GetVertexDefByType(0);
  EXPECT_EQ(user_def.Name(), "user");
  EXPECT_EQ(user_def.AttrTypes().size(), 4);
  auto& vertex_def_map = schema.VertexDefMap();
  EXPECT_EQ(vertex_def_map.at(1).Name(), "item");
  EXPECT_EQ(vertex_def_map.at(1).AttrTypes().at(3), 4);

  EXPECT_EQ(schema.EdgeDefNum(), 3);
  auto& buy_def = schema.GetEdgeDefByType(3);
  EXPECT_EQ(buy_def.Name(), "buy");
  EXPECT_EQ(buy_def.AttrTypes().at(0), 0);
  auto& edge_def_map = schema.EdgeDefMap();
  EXPECT_EQ(edge_def_map.at(2).Name(), "click");
  EXPECT_EQ(edge_def_map.at(4).Name(), "knows");
  EXPECT_EQ(edge_def_map.at(4).AttrTypes().size(), 4);

  EXPECT_EQ(schema.EdgeRelationDefNum(), 3);
  auto& edge_relation_defs = schema.EdgeRelationDefs();
  EXPECT_EQ(edge_relation_defs.at(0).Type(), 2);
  EXPECT_EQ(edge_relation_defs.at(1).SrcType(), 0);
  EXPECT_EQ(edge_relation_defs.at(2).DstType(), 0);

  UninitGoogleLogging();
}
