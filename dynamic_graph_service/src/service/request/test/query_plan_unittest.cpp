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

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include "gtest/gtest.h"

#include "core/execution/dag.h"
#include "generated/fbs/query_plan_generated.h"

using namespace dgs;
using namespace dgs::execution;

TEST(QueryPlan, Construct) {
  std::string schemafile;
  std::string jsonfile;

  bool ok = flatbuffers::LoadFile(
    "../../fbs/query_plan.fbs", false, &schemafile);
  EXPECT_TRUE(ok);
  ok = flatbuffers::LoadFile(
    "../../conf/ut/query_plan.ut.json", false, &jsonfile);
  EXPECT_TRUE(ok);

  flatbuffers::Parser parser;
  const char* include_paths[] = { "../../fbs/" };
  ok = parser.Parse(schemafile.c_str(), include_paths);
  EXPECT_TRUE(ok);
  ok = parser.Parse(jsonfile.c_str());
  EXPECT_TRUE(ok);

  uint8_t* buf = parser.builder_.GetBufferPointer();
  auto query_plan_rep = GetQueryPlanRep(buf);
  EXPECT_EQ(query_plan_rep->plan_nodes()->size(), 4);

  Dag dag(query_plan_rep);
  EXPECT_EQ(dag.num_nodes(), 4);
  EXPECT_EQ(dag.num_edges(), 5);

  auto root = dag.GetNode(0);
  auto node1 = dag.GetNode(1);
  EXPECT_EQ(root->in_edges().size(), 0);
  EXPECT_EQ(root->out_edges().size(), 2);
  EXPECT_EQ(root->out_edges()[0]->src_output(), 0);
  EXPECT_EQ(root->out_edges()[0]->dst_input(), 0);
  EXPECT_EQ(root->out_edges()[0]->src(), root);
  EXPECT_EQ(root->out_edges()[0]->dst(), node1);
}