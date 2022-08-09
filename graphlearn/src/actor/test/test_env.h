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

#ifndef GRAPHLEARN_ACTOR_TEST_ENV_H_
#define GRAPHLEARN_ACTOR_TEST_ENV_H_

#include <cstdint>
#include <fstream>
#include <string>

#include "core/io/element_value.h"
#include "include/config.h"
#include "include/server.h"

namespace graphlearn {
namespace act {

class TestEnv {
public:
  enum DatasetCode {
    kValid            = 0,
    kInvalidAttrType  = 1,
    kRedundantAttr    = 2,
    kMissingAttr      = 3,
    kInvalidDelimeter = 4
  };
public:
  explicit TestEnv(uint32_t load_batch_size = 10240,
                   uint32_t num_shards = 4,
                   uint32_t common_num = 40,
                   uint32_t remaining_factor = 5,
                   uint32_t edge_num = 1000,
                   uint32_t node_int_attr_num = 1,
                   uint32_t node_float_attr_num = 0,
                   DatasetCode code = DatasetCode::kValid);

  void Initialize();
  void Finalize();

private:
  void GenNodeTestData(const char* file_name);
  void GenEdgeTestData(const char* file_name);
  void MakeEdgeSource(io::EdgeSource* source,
                      const std::string& file_name,
                      const std::string& edge_type,
                      const std::string& src_type,
                      const std::string& dst_type);
  void MakeNodeSource(io::NodeSource* source,
                      const std::string& file_name,
                      const std::string& node_type);
  void WriteNodeRecord(std::ofstream *out, char* buffer,
                       int32_t buffer_size, int32_t id);
  void WriteFirstNode(std::ofstream *out, char* buffer,
                      int32_t buffer_size, int32_t id);
private:
  Server *server_ = nullptr;
  const uint32_t common_num_;
  const uint32_t remaining_factor_;
  const uint32_t num_shards_;
  const uint32_t edge_num_;
  const uint32_t load_batch_size_;
  const uint32_t node_int_attr_num_;
  const uint32_t node_float_attr_num_;
  const int32_t dataset_code_;
  const int32_t node_format_ = io::kAttributed;
  const int32_t edge_format_ = io::kWeighted;
  const std::string tracker_dir_ = "./tracker";
  const char* user_node_file_ = "user_attributed_node_file";
  const char* item_node_file_ = "item_attributed_node_file";
  const char* i2i_edge_file_ = "item_to_item_weighted_edge_file";
  const char* u2i_edge_file_ = "user_to_item_weighted_edge_file";
};

TestEnv::TestEnv(uint32_t load_batch_size,
                 uint32_t num_shards,
                 uint32_t common_num,
                 uint32_t remaining_factor,
                 uint32_t edge_num,
                 uint32_t node_int_attr_num,
                 uint32_t node_float_attr_num,
                 DatasetCode code)
  : load_batch_size_(load_batch_size),
    num_shards_(num_shards),
    common_num_(common_num),
    remaining_factor_(remaining_factor),
    edge_num_(edge_num),
    node_int_attr_num_(node_int_attr_num),
    node_float_attr_num_(node_float_attr_num),
    dataset_code_(code) {
}

void TestEnv::Initialize() {
  SetGlobalFlagEnableActor(1);
  SetGlobalFlagDeployMode(kLocal);
  SetGlobalFlagTrackerMode(kFileSystem);
  SetGlobalFlagActorLocalShardCount(num_shards_);
  SetGlobalFlagDataInitBatchSize(load_batch_size_);

  SetGlobalFlagTracker(tracker_dir_);
  std::string md_cmd = "mkdir -p " + tracker_dir_;
  ::system(md_cmd.data());

  GenNodeTestData(user_node_file_);
  GenNodeTestData(item_node_file_);
  GenEdgeTestData(i2i_edge_file_);
  GenEdgeTestData(u2i_edge_file_);

  std::vector<io::EdgeSource> edges(2);
  MakeEdgeSource(&edges[0], u2i_edge_file_,
    "click", "user", "item");
  MakeEdgeSource(&edges[1], i2i_edge_file_,
    "similar", "item", "item");

  std::vector<io::NodeSource> nodes(2);
  MakeNodeSource(&nodes[0], user_node_file_, "user");
  MakeNodeSource(&nodes[1], item_node_file_, "item");

  server_ = NewServer(0, 1, "127.0.0.1:8888", tracker_dir_);
  server_->Start();
  server_->Init(edges, nodes);
}

void TestEnv::Finalize() {
  server_->Stop();
  delete server_;

  std::string rm_cmd = "rm -rf " + tracker_dir_;
  ::system(rm_cmd.data());
}

void TestEnv::WriteFirstNode(std::ofstream *out, char* buffer,
                             int32_t buffer_size, int32_t id) {
  int offset = 0;
  offset += snprintf(buffer + offset, buffer_size - offset, "%d\t", id);
  int node_int_attr_num = node_int_attr_num_;
  assert(node_int_attr_num_ >= 1);
  if (dataset_code_ == DatasetCode::kMissingAttr) {
    node_int_attr_num = node_int_attr_num_ - 1;
  } else if (dataset_code_ == DatasetCode::kRedundantAttr) {
    node_int_attr_num = node_int_attr_num_ + 1;
  }

  for (int j = 0; j < node_int_attr_num; ++j) {
    if (dataset_code_ == DatasetCode::kInvalidAttrType) {
      offset += snprintf(buffer + offset, buffer_size - offset,
        "invalid:");
    } else if (dataset_code_ == DatasetCode::kInvalidDelimeter) {
      offset += snprintf(buffer + offset, buffer_size - offset, "%d;",
        id * 2);
    } else {
      offset += snprintf(buffer + offset, buffer_size - offset, "%d:",
        id * 2);
    }
  }
  for (int j = 0; j < node_float_attr_num_; ++j) {
    offset += snprintf(buffer + offset, buffer_size - offset, "%f:",
      float(id * 10));
  }

  buffer[offset - 1] = '\n';
  out->write(buffer, offset);
}

void TestEnv::WriteNodeRecord(std::ofstream *out, char* buffer,
                              int32_t buffer_size, int32_t id) {
  int offset = 0;
  offset += snprintf(buffer + offset, buffer_size - offset, "%d\t", id);
  for (int j = 0; j < node_int_attr_num_; ++j) {
    offset += snprintf(buffer + offset, buffer_size - offset, "%d:",
      id * 2);
  }
  for (int j = 0; j < node_float_attr_num_; ++j) {
    offset += snprintf(buffer + offset, buffer_size - offset, "%f:",
      float(id * 10));
  }

  buffer[offset - 1] = '\n';
  out->write(buffer, offset);
}

void TestEnv::GenNodeTestData(const char* file_name) {
  io::SideInfo info;
  info.format = node_format_;
  std::ofstream out(file_name);
  // write title
  const char* title = "node_id:int64\tattribute:string\n";
  out.write(title, strlen(title));
  int buffer_size = 640;
  char buffer[buffer_size];

  // write data
  WriteFirstNode(&out, buffer, buffer_size, 0);
  for (int32_t i = 1; i < common_num_ * num_shards_; i++) {
    WriteNodeRecord(&out, buffer, buffer_size, i);
  }

  for (int32_t i = 0; i < num_shards_; ++i) {
    auto start = common_num_ * num_shards_ + i;
    for (int32_t j = 0; j < remaining_factor_ * (i + 1); ++j) {
      WriteNodeRecord(&out, buffer, buffer_size, start);
      start += num_shards_;
    }
  }

  out.close();
}

void TestEnv::GenEdgeTestData(const char* file_name) {
  io::SideInfo info;
  info.format = edge_format_;
  int source_node_num = common_num_ * num_shards_;
  int dst_node_num = source_node_num;
  std::ofstream out(file_name);
  int iter = edge_num_ / source_node_num;
  // write title
  const char* title = "src_id:int64\tdst_id:int64\tedge_weight:float\n";
  out.write(title, strlen(title));
  // write user_to_item data
  int size = 0;
  int buffer_size = 640;
  char buffer[buffer_size];
  for (int32_t i = 0; i < source_node_num; ++i) {
    int src_user_id = i;
    for (int32_t j = 0; j < iter; ++j) {
      size = 0;
      int dst_item_id = rand() % dst_node_num;
      size = snprintf(buffer, sizeof(buffer), "%d\t%d\t%f\n",
                      src_user_id , dst_item_id, float(i));
      out.write(buffer, size);
    }
  }
  out.close();
}

void TestEnv::MakeEdgeSource(io::EdgeSource* source,
                             const std::string& file_name,
                             const std::string& edge_type,
                             const std::string& src_type,
                             const std::string& dst_type) {
  source->path = file_name;
  source->edge_type = edge_type;
  source->src_id_type = src_type;
  source->dst_id_type = dst_type;
  source->format = edge_format_;
  source->attr_info.ignore_invalid = false;
}

void TestEnv::MakeNodeSource(io::NodeSource* source,
                             const std::string& file_name,
                             const std::string& node_type) {
  source->path = file_name;
  source->id_type = node_type;
  source->format = node_format_;
  source->attr_info.ignore_invalid = false;
  if (node_format_ & io::kAttributed) {
    source->attr_info.delimiter = ":";
    source->attr_info.types.reserve(
      node_int_attr_num_ + node_float_attr_num_);
    for (int i = 0; i < node_int_attr_num_; i++) {
      source->attr_info.types.push_back(DataType::kInt32);
      source->attr_info.hash_buckets.push_back(0);
    }
    for (int i = 0; i < node_float_attr_num_; i++) {
      source->attr_info.types.push_back(DataType::kFloat);
      source->attr_info.hash_buckets.push_back(0);
    }
  }
}

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_TEST_ENV_H_
