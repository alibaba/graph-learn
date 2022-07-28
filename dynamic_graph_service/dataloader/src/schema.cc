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

#include "dataloader/schema.h"

#include <iostream>

namespace dgs {
namespace dataloader {

AttributeDef::AttributeDef(const boost::property_tree::ptree& node)
  : type_(node.get_child("type").get_value<AttributeType>()),
    name_(node.get_child("name").get_value<std::string>()) {
  auto value_type_name = node.get_child("value_type").get_value<std::string>();
  if (value_type_name == "INT32") {
    value_type_ = AttributeValueType::INT32;
  } else if (value_type_name == "INT32_LIST") {
    value_type_ = AttributeValueType::INT32_LIST;
  } else if (value_type_name == "INT64") {
    value_type_ = AttributeValueType::INT64;
  } else if (value_type_name == "INT64_LIST") {
    value_type_ = AttributeValueType::INT64_LIST;
  } else if (value_type_name == "FLOAT32") {
    value_type_ = AttributeValueType::FLOAT32;
  } else if (value_type_name == "FLOAT32_LIST") {
    value_type_ = AttributeValueType::FLOAT32_LIST;
  } else if (value_type_name == "FLOAT64") {
    value_type_ = AttributeValueType::FLOAT64;
  } else if (value_type_name == "FLOAT64_LIST") {
    value_type_ = AttributeValueType::FLOAT64_LIST;
  } else if (value_type_name == "STRING") {
    value_type_ = AttributeValueType::STRING;
  } else {
    std::cerr << "Unsupported attribute definition, "
              << "name: " << name_ << ", data type: " << value_type_name
              << std::endl;
    value_type_ = AttributeValueType::STRING;
  }
}

VertexDef::VertexDef(const boost::property_tree::ptree& node)
  : type_(node.get_child("vtype").get_value<VertexType>()),
    name_(node.get_child("name").get_value<std::string>()) {
  auto attr_types_node = node.get_child("attr_types");
  for (auto& iter : attr_types_node) {
    attr_types_.push_back(iter.second.get_value<AttributeType>());
  }
}

EdgeDef::EdgeDef(const boost::property_tree::ptree& node)
  : type_(node.get_child("etype").get_value<EdgeType>()),
    name_(node.get_child("name").get_value<std::string>()) {
  auto attr_types_node = node.get_child("attr_types");
  for (auto& iter : attr_types_node) {
    attr_types_.push_back(iter.second.get_value<AttributeType>());
  }
}

EdgeRelationDef::EdgeRelationDef(const boost::property_tree::ptree& node)
  : etype_(node.get_child("etype").get_value<EdgeType>()),
    src_type_(node.get_child("src_vtype").get_value<VertexType>()),
    dst_type_(node.get_child("dst_vtype").get_value<VertexType>()) {
}

void Schema::Init(const std::string& json) {
  std::stringstream ss(json);
  boost::property_tree::ptree ptree;
  boost::property_tree::read_json(ss, ptree);
  Init(ptree);
}

void Schema::Init(const boost::property_tree::ptree& node) {
  auto& attr_defs_node = node.get_child("attr_defs");
  for (auto& iter : attr_defs_node) {
    AttributeDef def(iter.second);
    type_to_attr_def_.emplace(def.Type(), def);
    name_to_attr_def_.emplace(def.Name(), def);
  }

  auto& vertex_defs_node = node.get_child("vertex_defs");
  for (auto& iter : vertex_defs_node) {
    VertexDef def(iter.second);
    type_to_vertex_def_.emplace(def.Type(), def);
    name_to_vertex_def_.emplace(def.Name(), def);
  }

  auto& edge_defs_node = node.get_child("edge_defs");
  for (auto& iter : edge_defs_node) {
    EdgeDef def(iter.second);
    type_to_edge_def_.emplace(def.Type(), def);
    name_to_edge_def_.emplace(def.Name(), def);
  }

  auto& edge_relation_defs_node = node.get_child("edge_relation_defs");
  for (auto& iter : edge_relation_defs_node) {
    edge_relation_defs_.emplace_back(iter.second);
  }
}

}  // namespace dataloader
}  // namespace dgs
