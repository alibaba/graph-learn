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

#ifndef DATALOADER_SCHEMA_H_
#define DATALOADER_SCHEMA_H_

#include <unordered_map>
#include <vector>

#include "dataloader/typedefs.h"
#include "dataloader/fbs/schema_generated.h"

namespace dgs {
namespace dataloader {

class AttributeDef {
public:
  explicit AttributeDef(const AttributeDefRep* rep);
  ~AttributeDef() = default;

  AttributeDef(const AttributeDef&) = default;
  AttributeDef& operator=(const AttributeDef&) = default;
  AttributeDef(AttributeDef&&) = default;
  AttributeDef& operator=(AttributeDef&&) = default;

  AttributeType Type() const {
    return type_;
  }

  const std::string& Name() const {
    return name_;
  }

  AttributeValueType ValueType() const {
    return value_type_;
  }

private:
  AttributeType type_;
  std::string name_;
  AttributeValueType value_type_;
};

class VertexDef {
public:
  explicit VertexDef(const VertexDefRep* rep);
  ~VertexDef() = default;

  VertexType Type() const {
    return type_;
  }

  const std::string& Name() const {
    return name_;
  }

  const std::vector<AttributeType>& AttrTypes() const {
    return attr_types_;
  }

private:
  VertexType type_;
  std::string name_;
  std::vector<AttributeType> attr_types_;
};

class EdgeDef {
public:
  explicit EdgeDef(const EdgeDefRep* rep);
  ~EdgeDef() = default;

  EdgeType Type() const {
    return type_;
  }

  const std::string& Name() const {
    return name_;
  }

  const std::vector<AttributeType>& AttrTypes() const {
    return attr_types_;
  }

private:
  EdgeType type_;
  std::string name_;
  std::vector<AttributeType> attr_types_;
};

class EdgeRelationDef {
public:
  explicit EdgeRelationDef(const EdgeRelationDefRep* rep);
  ~EdgeRelationDef() = default;

  EdgeType Type() const {
    return etype_;
  }

  VertexType SrcType() const {
    return src_type_;
  }

  VertexType DstType() const {
    return dst_type_;
  }

private:
  EdgeType etype_;
  VertexType src_type_;
  VertexType dst_type_;
};

class Schema {
public:
  static Schema& GetInstance() {
    static Schema instance;
    return instance;
  }

  bool Init();
  bool Init(const std::string& schema_json_file, const std::string& fbs_file,
            const std::vector<std::string>& fbs_include_paths);
  void Init(const SchemaRep* rep);

  size_t AttrDefNum() const {
    return type_to_attr_def_.size();
  }

  const AttributeDef& GetAttrDefByType(AttributeType type) const {
    return type_to_attr_def_.at(type);
  }

  const AttributeDef& GetAttrDefByName(const std::string& name) const {
    return name_to_attr_def_.at(name);
  }

  const std::unordered_map<AttributeType, AttributeDef>& AttrDefMap() const {
    return type_to_attr_def_;
  }

  size_t VertexDefNum() const {
    return type_to_vertex_def_.size();
  }

  const VertexDef& GetVertexDefByType(VertexType type) const {
    return type_to_vertex_def_.at(type);
  }

  const VertexDef& GetVertexDefByName(const std::string& name) const {
    return name_to_vertex_def_.at(name);
  }

  const std::unordered_map<VertexType, VertexDef>& VertexDefMap() const {
    return type_to_vertex_def_;
  }

  size_t EdgeDefNum() const {
    return type_to_edge_def_.size();
  }

  const EdgeDef& GetEdgeDefByType(EdgeType type) const {
    return type_to_edge_def_.at(type);
  }

  const EdgeDef& GetEdgeDefByName(const std::string& name) const {
    return name_to_edge_def_.at(name);
  }

  const std::unordered_map<EdgeType, EdgeDef>& EdgeDefMap() const {
    return type_to_edge_def_;
  }

  size_t EdgeRelationDefNum() const {
    return edge_relation_defs_.size();
  }

  const std::vector<EdgeRelationDef>& EdgeRelationDefs() const {
    return edge_relation_defs_;
  }

private:
  Schema() = default;
  ~Schema() = default;

private:
  std::unordered_map<AttributeType, AttributeDef> type_to_attr_def_;
  std::unordered_map<std::string, AttributeDef> name_to_attr_def_;
  std::unordered_map<VertexType, VertexDef> type_to_vertex_def_;
  std::unordered_map<std::string, VertexDef> name_to_vertex_def_;
  std::unordered_map<EdgeType, EdgeDef> type_to_edge_def_;
  std::unordered_map<std::string, EdgeDef> name_to_edge_def_;
  std::vector<EdgeRelationDef> edge_relation_defs_;
};

}  // namespace dataloader
}  // namespace dgs

#endif // DATALOADER_SCHEMA_H_
