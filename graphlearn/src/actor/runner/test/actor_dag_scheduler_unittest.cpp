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

#include <thread>

#include "gtest/gtest.h"
#include "google/protobuf/text_format.h"

#include "actor/test/test_env.h"
#include "core/dag/tape.h"
#include "core/runner/dag_scheduler.h"
#include "include/config.h"
#include "platform/env.h"
#include "platform/protobuf.h"

#include "generated/proto/dag.pb.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

class ActorDagSchedulerTest : public ::testing::Test {
public:
  ActorDagSchedulerTest() = default;
  ~ActorDagSchedulerTest() override = default;

protected:
  void SetUp() override {
    env_.Initialize();
  }

  void TearDown() override {
    env_.Finalize();
  }

protected:
  act::TestEnv env_;
};

TEST_F(ActorDagSchedulerTest, GetNodes) {
  std::string dag_content =
    "nodes { \n"
      "id: 1 \n"
      "op_name: \"GetNodes\" \n"
      "params { \n"
        "name: \"nf\" \n"
        "length: 1 \n"
        "int32_values: 2 \n"
      "} \n"
      "params { \n"
        "name: \"nt\" \n"
        "dtype: 4 \n"
        "length: 1 \n"
        "string_values: \"user\" \n"
      "} \n"
      "params { \n"
        "name: \"ep\" \n"
        "length: 1 \n"
        "int32_values: 2147483647 \n"
      "} \n"
      "params { \n"
        "name: \"bs\" \n"
        "length: 1 \n"
        "int32_values: 20 \n"
      "} \n"
      "params { \n"
        "name: \"str\" \n"
        "dtype: 4 \n"
        "length: 1 \n"
        "string_values: \"by_order\" \n"
      "} \n"
      "out_edges { \n"
        "id: 1 \n"
        "src_output: \"nid\" \n"
        "dst_input: \"ids\" \n"
      "} \n"
    "} \n"
    "nodes { \n"
      "id: 2 \n"
      "op_name: \"Sink\" \n"
      "in_edges { \n"
        "id: 1 \n"
        "src_output: \"nid\" \n"
        "dst_input: \"ids\" \n"
      "} \n"
    "}";

  DagDef def;
  Dag* dag = nullptr;
  PB_NAMESPACE::TextFormat::ParseFromString(dag_content, &def);
  Status s = DagFactory::GetInstance()->Create(def, &dag);
  EXPECT_TRUE(s.ok());

  DagScheduler::Take(Env::Default(), dag);
  TapeStorePtr store = GetTapeStore(dag->Id());
  Tape* tape = nullptr;

  for (int32_t idx = 0; idx < 10; ++idx) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    tape = store->WaitAndPop(GLOBAL_FLAG(ClientId));
    EXPECT_EQ(tape->Id(), idx);
    EXPECT_EQ(tape->Epoch(), 0);
    EXPECT_TRUE(tape->IsReady());
    EXPECT_EQ(tape->Size(), 2);
    auto& record = tape->Retrieval(1);
    EXPECT_EQ(record.at("nid").Size(), 20);
    delete tape;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  tape = store->WaitAndPop(GLOBAL_FLAG(ClientId));
  EXPECT_EQ(tape->Id(), 10);
  EXPECT_EQ(tape->Epoch(), 0);
  EXPECT_TRUE(tape->IsReady());
  EXPECT_EQ(tape->Size(), 2);
  auto& record1 = tape->Retrieval(1);
  EXPECT_EQ(record1.at("nid").Size(), 10);
  delete tape;

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  tape = store->WaitAndPop(GLOBAL_FLAG(ClientId));
  EXPECT_EQ(tape->Id(), 11);
  EXPECT_EQ(tape->Epoch(), 0);
  EXPECT_TRUE(tape->IsFaked());
  delete tape;

  for (int32_t idx = 12; idx < 22; ++idx) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    tape = store->WaitAndPop(GLOBAL_FLAG(ClientId));
    EXPECT_EQ(tape->Id(), idx);
    EXPECT_EQ(tape->Epoch(), 1);
    EXPECT_TRUE(tape->IsReady());
    EXPECT_EQ(tape->Size(), 2);
    auto& record = tape->Retrieval(1);
    EXPECT_EQ(record.at("nid").Size(), 20);
    delete tape;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  tape = store->WaitAndPop(GLOBAL_FLAG(ClientId));
  EXPECT_EQ(tape->Id(), 22);
  EXPECT_EQ(tape->Epoch(), 1);
  EXPECT_TRUE(tape->IsReady());
  EXPECT_EQ(tape->Size(), 2);
  auto& record2 = tape->Retrieval(1);
  EXPECT_EQ(record2.at("nid").Size(), 10);
  delete tape;

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  tape = store->WaitAndPop(GLOBAL_FLAG(ClientId));
  EXPECT_EQ(tape->Id(), 23);
  EXPECT_EQ(tape->Epoch(), 1);
  EXPECT_TRUE(tape->IsFaked());
  delete tape;
}
