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

#include "boost/asio.hpp"
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

#include "dataloader/service.h"

namespace bpo = boost::program_options;
namespace fs = boost::filesystem;
using namespace dgs::dataloader;

uint32_t loading_thread_num = 4;
std::string icbu_data_dir = "/tmp/icbu_data";

uint32_t max_loading_vertices_per_file = 200000000;
uint32_t max_loading_edges_per_file = 50000000;

const VertexType user_type = 1;
const VertexType item_type = 2;
const EdgeType u2i_type = 3;
const EdgeType i2i_type = 4;

const AttributeType timestamp_type = 1;
const AttributeValueType timestamp_value_type = dgs::dataloader::INT64;
const AttributeType weight_type = 2;
const AttributeValueType weight_value_type = dgs::dataloader::FLOAT32;
const AttributeType attributes_type = 3;
const AttributeValueType attributes_value_type = dgs::dataloader::STRING;

int64_t CurrentTimeInMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}

class IcbuLoader {
public:
  IcbuLoader();

  void LoadVertexFile(const std::string& file_path, VertexType vtype);
  void LoadEdgeFile(const std::string& file_path, EdgeType etype,
                    VertexType src_vtype, VertexType dst_vtype);

private:
  uint32_t FlushPending(uint32_t idx);

private:
  const uint32_t batch_size_;
  const uint32_t data_partition_num_;
  std::vector<AttrInfo> attrs_;
  BatchProducer producer_;
  std::vector<BatchBuilder> batch_builders_;
  std::vector<cppkafka::MessageBuilder> msg_builders_;
};

inline
IcbuLoader::IcbuLoader()
  : batch_size_(Options::GetInstance().output_batch_size),
    data_partition_num_(Options::GetInstance().data_partitions),
    attrs_(3),
    producer_() {
  attrs_[0].attr_type = timestamp_type;
  attrs_[0].value_type = timestamp_value_type;
  attrs_[1].attr_type = weight_type;
  attrs_[1].value_type = weight_value_type;
  attrs_[2].attr_type = attributes_type;
  attrs_[2].value_type = attributes_value_type;
  batch_builders_.reserve(data_partition_num_);
  msg_builders_.reserve(data_partition_num_);
  for (uint32_t i = 0; i < data_partition_num_; i++) {
    batch_builders_.emplace_back(i);
    msg_builders_.emplace_back(Options::GetInstance().output_kafka_topic);
    msg_builders_[i].partition(static_cast<int32_t>(Partitioner::GetInstance().GetKafkaPartitionId(i)));
  }
}

inline
void IcbuLoader::LoadVertexFile(const std::string& file_path, VertexType vtype) {
  std::ifstream infile(file_path);
  if (!infile.good()) {
    LOG(ERROR) << "Cannot open file of vertex table: " << file_path;
    return;
  }

  auto& partitioner = Partitioner::GetInstance();

  VertexId vid;
  int64_t timestamp;
  float weight;
  unsigned count = 0;
  uint32_t send_bytes = 0;
  int64_t start_time = CurrentTimeInMs();
  int64_t end_time;
  while (infile >> vid >> weight >> attrs_[2].value_bytes) {
    timestamp = CurrentTimeInMs();
    attrs_[0].value_bytes.assign(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
    attrs_[1].value_bytes.assign(reinterpret_cast<const char*>(&weight), sizeof(weight));
    auto data_pid = partitioner.GetDataPartitionId(vid);
    auto& bb = batch_builders_.at(data_pid);
    bb.AddVertexUpdate(vtype, vid, attrs_);
    if (bb.RecordNum() >= batch_size_) {
      send_bytes += FlushPending(data_pid);
    }
    count++;
    if (count % 1000000 == 0) {
      end_time = CurrentTimeInMs();
      double tp = ((double)(send_bytes) / 1024 / 1024) / ((double)(end_time - start_time) / 1000);
      LOG(INFO) << file_path + ": processed num: " << count << ", throughput: " << tp;
      start_time = CurrentTimeInMs();
      send_bytes = 0;
    }
    if (count >= max_loading_vertices_per_file) {
      break;
    }
  }
  // Flush remaining
  for (uint32_t i = 0; i < data_partition_num_; i++) {
    if (batch_builders_.at(i).RecordNum() > 0) {
      send_bytes += FlushPending(i);
    }
  }
  infile.close();
  LOG(INFO) << "Finished loading vertex table: " << file_path << ", loading record number: " << count;
}

void IcbuLoader::LoadEdgeFile(const std::string& file_path, EdgeType etype,
                              VertexType src_vtype, VertexType dst_vtype) {
  std::ifstream infile(file_path);
  if (!infile.good()) {
    LOG(ERROR) << "Cannot open file of edge table: " << file_path;
    return;
  }

  auto& partitioner = Partitioner::GetInstance();

  VertexId src_vid, dst_vid;
  int64_t timestamp;
  float weight;
  unsigned count = 0;
  uint32_t send_bytes = 0;
  int64_t start_time = CurrentTimeInMs();
  int64_t end_time;
  while (infile >> src_vid >> dst_vid >> weight >> attrs_[2].value_bytes) {
    timestamp = CurrentTimeInMs();
    attrs_[0].value_bytes.assign(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
    attrs_[1].value_bytes.assign(reinterpret_cast<const char*>(&weight), sizeof(weight));
    auto data_pid = partitioner.GetDataPartitionId(src_vid);
    auto& bb = batch_builders_.at(data_pid);
    bb.AddEdgeUpdate(etype, src_vtype, dst_vtype, src_vid, dst_vid, attrs_);
    if (bb.RecordNum() >= batch_size_) {
      send_bytes += FlushPending(data_pid);
    }

    count++;
    if (count % 1000000 == 0) {
      end_time = CurrentTimeInMs();
      double tp = ((double)(send_bytes) / 1024 / 1024) / ((double)(end_time - start_time) / 1000);
      LOG(INFO) << file_path + ": processed num: " << count << ", throughput: " << tp;
      start_time = CurrentTimeInMs();
      send_bytes = 0;
    }
    if (count >= max_loading_edges_per_file) {
      break;
    }
  }
  // Flush remaining
  for (uint32_t i = 0; i < data_partition_num_; i++) {
    if (batch_builders_.at(i).RecordNum() > 0) {
      send_bytes += FlushPending(i);
    }
  }
  infile.close();
  LOG(INFO) << "Finished loading edge table: " << file_path << ", loading record number: " << count;
}

inline
uint32_t IcbuLoader::FlushPending(uint32_t idx) {
  auto& bb = batch_builders_.at(idx);
  bb.Finish();
  auto& mb = msg_builders_.at(idx);
  mb.payload({bb.GetBufferPointer(), bb.GetBufferSize()});
  producer_.SyncProduce(mb);
  auto flushed_bytes = bb.GetBufferSize();
  bb.Clear();
  return flushed_bytes;
}

void LoadIcbuData(const std::string& data_dir) {
  auto start = CurrentTimeInMs();
  boost::asio::thread_pool pool(loading_thread_num);
  fs::path icbu_path(data_dir);
  fs::recursive_directory_iterator end_iter;
  for (fs::recursive_directory_iterator iter(icbu_path / "user"); iter != end_iter; iter++) {
    boost::asio::post(pool, [file_path = (*iter).path().string()] {
      IcbuLoader loader;
      loader.LoadVertexFile(file_path, user_type);
    });
  }
  for (fs::recursive_directory_iterator iter(icbu_path / "item"); iter != end_iter; iter++) {
    boost::asio::post(pool, [file_path = (*iter).path().string()] {
      IcbuLoader loader;
      loader.LoadVertexFile(file_path, item_type);
    });
  }
  fs::recursive_directory_iterator u2i_iter(icbu_path / "u2i");
  fs::recursive_directory_iterator i2i_iter(icbu_path / "i2i");
  while (u2i_iter != end_iter || i2i_iter != end_iter) {
    if (u2i_iter != end_iter) {
      boost::asio::post(pool, [file_path = (*u2i_iter).path().string()] {
        IcbuLoader loader;
        loader.LoadEdgeFile(file_path, u2i_type, user_type, item_type);
      });
      u2i_iter++;
    }
    if (i2i_iter != end_iter) {
      boost::asio::post(pool, [file_path = (*i2i_iter).path().string()] {
        IcbuLoader loader;
        loader.LoadEdgeFile(file_path, i2i_type, item_type, item_type);
      });
      i2i_iter++;
    }
  }
  pool.join();
  auto end = CurrentTimeInMs();
  LOG(INFO) << "Bulk loading finished, Time: " << (end - start) / 1000 << "s";
}

class IcbuLoadingService : public Service {
public:
  explicit IcbuLoadingService(const std::string& config_file, int32_t worker_id) : Service(config_file, worker_id) {}
  ~IcbuLoadingService() = default;

protected:
  void BulkLoad() override {
    LOG(INFO) << "Loading icbu data ...";
    LoadIcbuData(icbu_data_dir);
    LOG(INFO) << "Finish loading icbu data.";
  }

  void StreamingLoad() override {}
};

int main(int argc, char** argv) {
  bpo::options_description options("Icbu Data Loading Service Options");
  options.add_options()
  ("option-file,o", bpo::value<std::string>(), "dataloader option file")
  ("icbu-data-dir,d", bpo::value<std::string>(), "dir of icbu data")
  ("thread-num,t", bpo::value<uint32_t>(), "the thread number for loading")
  ("max-loading-num-per-vertex-file,v", bpo::value<uint32_t>(), "max loading number of vertices for each vertex file")
  ("max-loading-num-per-edge-file,e", bpo::value<uint32_t>(), "max loading number of edges for each edge file")
  ("log-to-console", "logs are written to standard error as well as to files.");
  bpo::variables_map vm;
  try {
    bpo::store(bpo::parse_command_line(argc, argv, options), vm);
  } catch (...) {
    std::cerr << "Undefined options in command line." << std::endl;
    return -1;
  }
  bpo::notify(vm);

  std::string option_file;
  if (vm.count("option-file")) {
    option_file = vm["option-file"].as<std::string>();
  } else {
    std::cerr << "The dataloader option file must be specified!" << std::endl;
    return -1;
  }

  if (vm.count("icbu-data-dir")) {
    icbu_data_dir = vm["icbu-data-dir"].as<std::string>();
  } else {
    std::cerr << "The icbu data directory must be specified!" << std::endl;
    return -1;
  }

  if (vm.count("thread-num")) {
    loading_thread_num = vm["thread-num"].as<uint32_t>();
  }

  if (vm.count("max-loading-num-per-vertex-file")) {
    max_loading_vertices_per_file = vm["max-loading-num-per-vertex-file"].as<uint32_t>();
  }
  if (vm.count("max-loading-num-per-edge-file")) {
    max_loading_edges_per_file = vm["max-loading-num-per-edge-file"].as<uint32_t>();
  }

  IcbuLoadingService service(option_file, 0);

  if (vm.count("log-to-console")) {
    FLAGS_alsologtostderr = true;
  }

  service.Run();
}
