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

#ifndef DGS_COMMON_SCHEMA_H_
#define DGS_COMMON_SCHEMA_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "common/typedefs.h"
#include "generated/fbs/schema_generated.h"

namespace dgs {

class AttributeDef {
public:
  explicit AttributeDef(const AttributeDefRep* rep);
  ~AttributeDef() = default;

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
    return vertex_defs_.size();
  }

  const VertexDef& GetVertexDefByType(VertexType type) const {
    return vertex_defs_.at(type);
  }

  const std::unordered_map<VertexType, VertexDef>& VertexDefMap() const {
    return vertex_defs_;
  }

  size_t EdgeDefNum() const {
    return edge_defs_.size();
  }

  const EdgeDef& GetEdgeDefByType(EdgeType type) const {
    return edge_defs_.at(type);
  }

  const std::unordered_map<EdgeType, EdgeDef>& EdgeDefMap() const {
    return edge_defs_;
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
  std::unordered_map<VertexType, VertexDef> vertex_defs_;
  std::unordered_map<EdgeType, EdgeDef> edge_defs_;
  std::vector<EdgeRelationDef> edge_relation_defs_;
};

}  // namespace dgs

#endif  // DGS_COMMON_SCHEMA_H_
