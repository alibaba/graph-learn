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

#include "dataloader/dataloader.h"

#include <curl/curl.h>
#include <iostream>

#include "boost/property_tree/json_parser.hpp"
#include "boost/property_tree/ptree.hpp"

namespace dgs {
namespace dataloader {

size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
  ((std::string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

void Initialize(const std::string& dgs_host) {
  CURL *curl;
  std::string res;
  curl = curl_easy_init();
  if (curl) {
    auto url = dgs_host + "/admin/init-info/dataloader";
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &res);
    auto s = curl_easy_perform(curl);
    if (s == CURLE_OK) {
      std::stringstream ss(res);
      boost::property_tree::ptree ptree;
      boost::property_tree::read_json(ss, ptree);

      auto& opts = Options::Get();

      // Init schema
      auto& schema_node = ptree.get_child("schema");
      Schema::Get().Init(schema_node);

      // Init downstream kafka info
      auto& kafka_node = ptree.get_child("downstream.kafka");
      std::vector<std::string> brokers;
      for (auto& iter: kafka_node.get_child("brokers")) {
        brokers.emplace_back(iter.second.get_value<std::string>());
      }
      opts.output_kafka_brokers = StrJoin(brokers, ",");
      opts.output_kafka_topic = kafka_node.get_child("topic").get_value<std::string>();
      opts.output_kafka_partitions = kafka_node.get_child("partitions").get_value<uint32_t>();

      // Init downstream partitioning info
      auto& partition_node = ptree.get_child("downstream.partition");
      opts.data_partitions = partition_node.get_child("data_partition_num").get_value<uint32_t>();
      std::vector<PartitionId> kafka_router;
      for (auto& iter : partition_node.get_child("kafka_router")) {
        kafka_router.push_back(iter.second.get_value<PartitionId>());
      }
      Partitioner::Get().Set(opts.data_partitions, std::move(kafka_router));

      std::cout << "*** Initialing Data Loading Components Successfully ***" << std::endl;
      std::cout << "-- graph schema: " << std::endl;
      boost::property_tree::write_json(std::cout, schema_node);
      std::cout << "-- downstream kafka brokers: " << opts.output_kafka_brokers << std::endl;
      std::cout << "-- downstream kafka topic: " << opts.output_kafka_topic << std::endl;
      std::cout << "-- downstream kafka partitions: " << opts.output_kafka_partitions << std::endl;
      std::cout << "-- downstream data partitions: " << opts.data_partitions << std::endl;
    } else {
      std::cerr << "Cannot get data loading init info: " << res << std::endl;
    }
    curl_easy_cleanup(curl);
  } else {
    std::cerr << "Cannot init curl environment." << std::endl;
  }
}

void SetBarrier(const std::string& dgs_host,
                const std::string& barrier_name,
                uint32_t dl_count,
                uint32_t dl_id) {
  CURL *curl;
  std::string res;
  curl = curl_easy_init();
  if (curl) {
    auto url = dgs_host + "/admin/barrier/set";
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    std::stringstream params;
    params << "name=" << barrier_name << "&count=" << dl_count << "&id=" << dl_id;
    auto params_str = params.str();
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, params_str.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, params_str.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &res);
    auto s = curl_easy_perform(curl);
    if (s == CURLE_OK) {
      std::cout << "Set barrier " << barrier_name << " on dataloader " << dl_id
                << ": " << res << std::endl;
    } else {
      std::cerr << "Cannot set barrier " << barrier_name << " on dataloader " << dl_id
                << ": " << res << std::endl;
    }
  } else {
    std::cerr << "Cannot init curl environment." << std::endl;
  }
}

BarrierStatus CheckBarrier(const std::string& dgs_host, const std::string& barrier_name) {
  CURL *curl;
  std::string res;
  curl = curl_easy_init();
  if (curl) {
    std::stringstream url_ss;
    url_ss << dgs_host << "/admin/barrier/status" << "?" << "name=" << barrier_name;
    curl_easy_setopt(curl, CURLOPT_URL, url_ss.str().c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &res);
    auto s = curl_easy_perform(curl);
    if (s == CURLE_OK) {
      if (res == "PRODUCED") {
        return BarrierStatus::PRODUCED;
      } else if (res == "SAMPLED") {
        return BarrierStatus::SAMPLED;
      } else if (res == "READY") {
        return BarrierStatus::READY;
      } else {
        return BarrierStatus::NOT_SET;
      }
    } else {
      std::cerr << "Cannot get status of barrier " << barrier_name << ": " << res << std::endl;
    }
  } else {
    std::cerr << "Cannot init curl environment." << std::endl;
  }
  return BarrierStatus::NOT_SET;
}

}  // namespace dataloader
}  // namespace dgs
