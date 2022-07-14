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

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

#include "dataloader/logging.h"
#include "dataloader/options.h"

namespace dgs {
namespace dataloader {

AttributeDef::AttributeDef(const boost::property_tree::ptree& node)
  : type_(node.get_child("type").get_value<AttributeType>()),
    name_(node.get_child("name").get_value<std::string>()) {
  auto value_type_name = node.get_child("value_type").get_value<std::string>();
  if (value_type_name == "BOOL") {
    value_type_ = AttributeValueType::BOOL;
  } else if (value_type_name == "CHAR") {
    value_type_ = AttributeValueType::CHAR;
  } else if (value_type_name == "INT16") {
    value_type_ = AttributeValueType::INT16;
  } else if (value_type_name == "INT32") {
    value_type_ = AttributeValueType::INT32;
  } else if (value_type_name == "INT64") {
    value_type_ = AttributeValueType::INT64;
  } else if (value_type_name == "FLOAT32") {
    value_type_ = AttributeValueType::FLOAT32;
  } else if (value_type_name == "FLOAT64") {
    value_type_ = AttributeValueType::FLOAT64;
  } else if (value_type_name == "STRING") {
    value_type_ = AttributeValueType::STRING;
  } else if (value_type_name == "BYTES") {
    value_type_ = AttributeValueType::BYTES;
  } else {
    LOG(ERROR) << "Unsupported attribute definition, "
               << "name: " << name_ << ", data type: " << value_type_name;
    value_type_ = AttributeValueType::STRING;
  }
}

VertexDef::VertexDef(const boost::property_tree::ptree& node)
  : type_(node.get_child("vtype").get_value<VertexType>()),
    name_(node.get_child("name").get_value<std::string>()) {
  auto
  auto attr_num = rep->attr_types()->size();
  attr_types_.reserve(attr_num);
  for (uint32_t i = 0; i < attr_num; i++) {
    attr_types_.emplace_back(rep->attr_types()->Get(i));
  }
}

EdgeDef::EdgeDef(const EdgeDefRep* rep)
  : type_(rep->etype()),
    name_(rep->name()->str()) {
  auto attr_num = rep->attr_types()->size();
  attr_types_.reserve(attr_num);
  for (uint32_t i = 0; i < attr_num; i++) {
    attr_types_.emplace_back(rep->attr_types()->Get(i));
  }
}

EdgeRelationDef::EdgeRelationDef(const EdgeRelationDefRep* rep)
  : etype_(rep->etype()),
    src_type_(rep->src_vtype()),
    dst_type_(rep->dst_vtype()) {
}

bool Schema::Init() {
  auto& opts = Options::GetInstance();
  return Init(opts.schema_file, opts.fbs_file_dir + "/schema.fbs", { opts.fbs_file_dir });
}

bool Schema::Init(const std::string& schema_json_file,
                  const std::string& fbs_file,
                  const std::vector<std::string>& fbs_include_paths) {
  std::string schema_buf;
  std::string json_buf;
  if (!flatbuffers::LoadFile(fbs_file.c_str(), false, &schema_buf)) {
    LOG(ERROR) << "Loading failed with schema fbs file: " << fbs_file;
    return false;
  }
  if (!flatbuffers::LoadFile(schema_json_file.c_str(), false, &json_buf)) {
    LOG(ERROR) << "Loading failed with schema json file: " << schema_json_file;
    return false;
  }
  std::vector<const char*> includes;
  std::transform(fbs_include_paths.begin(),
                 fbs_include_paths.end(),
                 std::back_inserter(includes),
                 [] (const std::string& str) { return str.c_str(); });
  flatbuffers::Parser parser;
  if (!parser.Parse(schema_buf.c_str(), includes.data())) {
    LOG(ERROR) << "Parsing failed with schema fbs file: " << fbs_file;
    return false;
  }
  if (!parser.Parse(json_buf.c_str())) {
    LOG(ERROR) << "Parsing failed with schema json file:  " << schema_json_file;
    return false;
  }
  Init(GetSchemaRep(parser.builder_.GetBufferPointer()));
  return true;
}

void Schema::Init(const dgs::SchemaRep* rep) {
  auto attr_num = rep->attr_defs()->size();
  type_to_attr_def_.reserve(attr_num);
  name_to_attr_def_.reserve(attr_num);
  for (uint32_t i = 0; i < attr_num; i++) {
    auto* attr_rep = rep->attr_defs()->Get(i);
    type_to_attr_def_.emplace(attr_rep->type(), AttributeDef{attr_rep});
    name_to_attr_def_.emplace(attr_rep->name()->str(), AttributeDef{attr_rep});
  }

  auto vertex_num = rep->vertex_defs()->size();
  type_to_vertex_def_.reserve(vertex_num);
  name_to_vertex_def_.reserve(vertex_num);
  for (uint32_t i = 0; i < vertex_num; i++) {
    auto* vertex_rep = rep->vertex_defs()->Get(i);
    type_to_vertex_def_.emplace(vertex_rep->vtype(), VertexDef{vertex_rep});
    name_to_vertex_def_.emplace(vertex_rep->name()->str(), VertexDef{vertex_rep});
  }

  auto edge_num = rep->edge_defs()->size();
  type_to_edge_def_.reserve(edge_num);
  name_to_edge_def_.reserve(edge_num);
  for (uint32_t i = 0; i < edge_num; i++) {
    auto* edge_rep = rep->edge_defs()->Get(i);
    type_to_edge_def_.emplace(edge_rep->etype(), EdgeDef{edge_rep});
    name_to_edge_def_.emplace(edge_rep->name()->str(), EdgeDef{edge_rep});
  }

  auto edge_relation_num = rep->edge_relation_defs()->size();
  edge_relation_defs_.reserve(edge_relation_num);
  for (uint32_t i = 0; i < edge_relation_num; i++) {
    edge_relation_defs_.emplace_back(rep->edge_relation_defs()->Get(i));
  }
}

}  // namespace dataloader
}  // namespace dgs
