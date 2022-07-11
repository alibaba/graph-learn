/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "boost/program_options.hpp"
#include "lgraph/client/graph_client.h"

namespace bpo = boost::program_options;

uint32_t extra_attr_size = 100;
uint32_t batch_size = 64;

int main(int argc, char** argv) {
  bpo::options_description options("GraphScope Record Generator Options");
  options.add_options()
    ("rpc-target,t", bpo::value<std::string>(), "the rpc server of graphscope-store service")
    ("schema-file,s", bpo::value<std::string>(), "the json schema file of graphscope-store")
    ("attr-bytes,a", bpo::value<uint32_t>(), "extra record attribute bytes")
    ("batch-size,b", bpo::value<uint32_t>(), "record batch size");
  bpo::variables_map vm;
  try {
    bpo::store(bpo::parse_command_line(argc, argv, options), vm);
  } catch (...) {
    std::cerr << "Undefined options in command line." << std::endl;
    return -1;
  }
  bpo::notify(vm);

  if (!vm.count("rpc-target")) {
    std::cerr << "Missing rpc target!" << std::endl;
    return -1;
  }
  std::string target = vm["rpc-target"].as<std::string>();

  if (!vm.count("schema-file")) {
    std::cerr << "Missing schema file!" << std::endl;
    return -1;
  }
  std::string schema_file = vm["schema-file"].as<std::string>();

  if (vm.count("attr-bytes")) {
    extra_attr_size = vm["attr-bytes"].as<uint32_t>();
  }

  if (vm.count("batch-size")) {
    batch_size = vm["batch-size"].as<uint32_t>();
  }

  lgraph::client::GraphClient client(target);
  uint32_t client_idx = 0;
  client.LoadJsonSchema(schema_file);

  std::string extra_attr;
  extra_attr.resize(extra_attr_size, 'a');
  std::unordered_map<std::string, std::string> props = {
      {"id", ""},
      {"timestamp", ""},
      {"weight", "1.0"},
      {"label", "10"},
      {"attr", extra_attr}
  };

  lgraph::client::BatchBuilder builder;
  lgraph::SnapshotId snapshot_id = 0;

  // Add vertices
  const std::string user_label = "user";
  std::string vid;
  for (int32_t i = 1; i <= 10; i++) {
    vid = std::to_string(i);
    props["id"] = vid;
    props["timestamp"] = vid;
    builder.AddVertex(user_label, vid, props);
    if (builder.Size() >= batch_size) {
      auto client_id = "0-" + std::to_string(client_idx++);
      snapshot_id = client.BatchWrite(builder.AsRequest(client_id));
    }
    if (i <= 5) {
      for (int32_t j = 1; j <= 9; j++) {
        vid = std::to_string(i * 10 + j);
        props["id"] = vid;
        props["timestamp"] = vid;
        builder.AddVertex(user_label, vid, props);
        if (builder.Size() >= batch_size) {
          auto client_id = "0-" + std::to_string(client_idx++);
          snapshot_id = client.BatchWrite(builder.AsRequest(client_id));
        }
      }
    }
  }

  // Construct the 1-hop neighbours for vertex v_1 ~ v_5.
  const std::string edge_label = "knows";
  int64_t edge_inner_id = 0;
  for (int32_t i = 1; i <= 5; i++) {
    int32_t src_vid = i;
    for (int32_t j = 1; j <= 9; j++) {
      int32_t dst_vid = src_vid * 10 + j;
      props["id"] = std::to_string(edge_inner_id);
      props["timestamp"] = std::to_string(dst_vid);
      builder.AddEdge(edge_label, edge_inner_id,
                      user_label, std::to_string(src_vid),
                      user_label, std::to_string(dst_vid), props);
      edge_inner_id++;
      if (builder.Size() >= batch_size) {
        auto client_id = "0-" + std::to_string(client_idx++);
        snapshot_id = client.BatchWrite(builder.AsRequest(client_id));
      }
    }
  }

  // Commit remaining and flush
  if (builder.Size() > 0) {
    auto client_id = "0-" + std::to_string(client_idx++);
    snapshot_id = client.BatchWrite(builder.AsRequest(client_id));
  }
  while (!client.RemoteFlush(snapshot_id, 1000)) {}

  std::this_thread::sleep_for(std::chrono::seconds(5));

  // Create backup and restore
  auto backup_id = client.CreateNewBackup();
  do {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  } while (!client.VerifyBackup(backup_id));
  client.RestoreFromBackup(backup_id, "/tmp/maxgraph_data/restored/meta", "/tmp/maxgraph_data/restored/store");

  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Construct the 1-hop neighbours for vertex v_6 ~ v_10 and shared 2-hop neighbours.
  for (int32_t i = 6; i <= 10; i++) {
    int32_t hop_1_src_vid = i;
    // Hop-1 neighbours
    for (int32_t j = 1; j <= 9; j++) {
      int32_t hop_1_dst_vid = (hop_1_src_vid - 5) * 10 + j;
      props["id"] = std::to_string(edge_inner_id);
      props["timestamp"] = std::to_string(hop_1_dst_vid);
      builder.AddEdge(edge_label, edge_inner_id,
                      user_label, std::to_string(hop_1_src_vid),
                      user_label, std::to_string(hop_1_dst_vid), props);
      edge_inner_id++;
      if (builder.Size() >= batch_size) {
        auto client_id = "0-" + std::to_string(client_idx++);
        snapshot_id = client.BatchWrite(builder.AsRequest(client_id));
      }
      // Hop-2 neighbours
      for (int32_t k = 1; k <= 9; k++) {
        int32_t hop_2_src_vid = hop_1_dst_vid;
        int32_t hop_2_dst_vid = hop_2_src_vid * 10 + k;
        props["id"] = std::to_string(edge_inner_id);
        props["timestamp"] = std::to_string(hop_2_dst_vid);
        builder.AddEdge(edge_label, edge_inner_id,
                        user_label, std::to_string(hop_2_src_vid),
                        user_label, std::to_string(hop_2_dst_vid), props);
        edge_inner_id++;
        if (builder.Size() >= batch_size) {
          auto client_id = "0-" + std::to_string(client_idx++);
          snapshot_id = client.BatchWrite(builder.AsRequest(client_id));
        }
      }
    }
  }

  // Update the timestamp for v_10 for 10 times.
  for (int32_t i = 1; i < 10; i++) {
    props["id"] = "10";
    // The initial timestamp for v_10 is 10
    props["timestamp"] = std::to_string(10 + i);
    builder.AddVertex(user_label, "10", props);
    if (builder.Size() >= batch_size) {
      auto client_id = "0-" + std::to_string(client_idx++);
      snapshot_id = client.BatchWrite(builder.AsRequest(client_id));
    }
  }

  // Commit remaining and flush
  if (builder.Size() > 0) {
    auto client_id = "0-" + std::to_string(client_idx++);
    snapshot_id = client.BatchWrite(builder.AsRequest(client_id));
  }
  while (!client.RemoteFlush(snapshot_id, 1000)) {}

  std::this_thread::sleep_for(std::chrono::seconds(2));
  return 0;
}