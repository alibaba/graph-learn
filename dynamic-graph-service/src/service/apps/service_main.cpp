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

#include "boost/program_options.hpp"

#include "common/log.h"
#include "service/service.h"

namespace bpo = boost::program_options;

int main(int argc, char* argv[]) {
  bpo::options_description desc("Usage:");
  desc.add_options()
    ("help", "Display this help message")
    ("config-file", bpo::value<std::string>(), "service configuration file")
    ("worker-id", bpo::value<uint32_t>(), "service worker id")
    ("log-to-console", "logs are written to standard error as well as to files.");

  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  if (!vm.count("config-file")) {
    std::cerr << "Service configuration file is not provided. "
              << "Using `--help` for more info" << std::endl;
    return -1;
  }
  std::string config_file = vm["config-file"].as<std::string>();

  if (!vm.count("worker-id")) {
    std::cerr << "Service worker id is not provided. "
              << "Using `--help` for more info" << std::endl;
    return -1;
  }
  uint32_t worker_id = vm["worker-id"].as<uint32_t>();

  dgs::Service service(config_file, worker_id);

  if (vm.count("log-to-console")) {
    FLAGS_alsologtostderr = true;
  }

  service.Run();
}
