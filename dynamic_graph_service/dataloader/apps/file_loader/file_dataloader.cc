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

#include <iostream>

#include "boost/program_options.hpp"
#include "dataloader/dataloader.h"

#include "loader.h"

namespace bpo = boost::program_options;
using namespace dgs::dataloader::file;

int main(int argc, char** argv) {
  bpo::options_description options("File Dataloader Options");
  options.add_options()
  ("dgs-service-host", bpo::value<std::string>(), "dgs service host")
  ("pattern-file", bpo::value<std::string>(), "pattern definition file of records")
  ("data-file", bpo::value<std::string>(), "data file of records")
  ("delimiter", bpo::value<char>()->default_value('&'), "delimiter of file contents")
  ("batch-size", bpo::value<uint32_t>()->default_value(16), "output batch size");
  bpo::variables_map vm;
  try {
    bpo::store(bpo::parse_command_line(argc, argv, options), vm);
  } catch (...) {
    std::cerr << "Undefined options in command line." << std::endl;
    return -1;
  }
  bpo::notify(vm);

  if (vm.count("help")) {
    std::cout << options << std::endl;
    return 0;
  }

  if (!vm.count("dgs-service-host")) {
    std::cerr << "The dgs service host must be specified!" << std::endl;
    return -1;
  }
  std::string dgs_host = vm["dgs-service-host"].as<std::string>();

  if (!vm.count("pattern-file")) {
    std::cerr << "The pattern definition file must be specified!" << std::endl;
    return -1;
  }
  std::string pattern_file = vm["pattern-file"].as<std::string>();

  if (!vm.count("data-file")) {
    std::cerr << "The data file to load must be specified!" << std::endl;
    return -1;
  }
  std::string data_file = vm["data-file"].as<std::string>();

  delimiter = vm["delimiter"].as<char>();
  batch_size = vm["batch-size"].as<uint32_t>();

  dgs::dataloader::Initialize(dgs_host);

  FileLoader loader(pattern_file);
  loader.Load(data_file);

  return 0;
}