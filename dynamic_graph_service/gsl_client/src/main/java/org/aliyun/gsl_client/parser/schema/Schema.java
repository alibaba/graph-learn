package org.aliyun.gsl_client.parser.schema;

import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.status.ErrorCode;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class Schema {
    private Map<String, AttrDef> nameToAttrDef;
    private Map<Integer, AttrDef> idToAttrDef;
    private Map<String, TypeDef> nameToTypeDef;
    private Map<Integer, TypeDef> idToTypeDef;
    private Map<String, EdgeRelation> edgeNameToRelation;
    private Map<Integer, EdgeRelation> edgeIdToRelation;

    public AttrDef getAttrDef(String attrName) throws UserException {
        if (!this.nameToAttrDef.containsKey(attrName)) {
            throw new UserException(ErrorCode.PAESE_ERROR, "Missing definition of attr name: " + attrName);
        }
        return this.nameToAttrDef.get(attrName);
    }

    public AttrDef getAttrDef(int attrTypeId) throws UserException {
        if (!this.idToAttrDef.containsKey(attrTypeId)) {
            throw new UserException(ErrorCode.PAESE_ERROR, "Missing definition of attr type: " + attrTypeId);
        }
        return this.idToAttrDef.get(attrTypeId);
    }

    public List<AttrDef> getAllAttrDefinitions() {
        return new ArrayList<>(this.nameToAttrDef.values());
    }

    public TypeDef getTypeDef(String typeName) throws UserException {
        if (!this.nameToTypeDef.containsKey(typeName)) {
            throw new UserException(ErrorCode.PAESE_ERROR, "Missing definition of type name: " + typeName);
        }
        return this.nameToTypeDef.get(typeName);
    }

    public TypeDef getTypeDef(int typeId) throws UserException {
        if (!this.idToTypeDef.containsKey(typeId)) {
            throw new UserException(ErrorCode.PAESE_ERROR, "Missing definition of type: " + typeId);
        }
        return this.idToTypeDef.get(typeId);
    }

    public List<TypeDef> getAllTypeDefinitions() {
        return new ArrayList<>(this.idToTypeDef.values());
    }

    public EdgeRelation getRelation(String edgeName) throws UserException {
        if (!this.edgeNameToRelation.containsKey(edgeName)) {
            throw new UserException(ErrorCode.PAESE_ERROR, "Missing relation of edge name: " + edgeName);
        }
        return this.edgeNameToRelation.get(edgeName);
    }

    public EdgeRelation getRelation(int edgeTypeId) throws UserException {
        if (!this.edgeIdToRelation.containsKey(edgeTypeId)) {
            throw new UserException(ErrorCode.PAESE_ERROR, "Missing relation of edge type: " + edgeTypeId);
        }
        return this.edgeIdToRelation.get(edgeTypeId);
    }

    public List<EdgeRelation> getAllEdgeRelations() {
        return new ArrayList<>(this.edgeIdToRelation.values());
    }

    public static Schema parseFrom(String jsonFile) throws UserException, IOException {
        byte[] content = Files.readAllBytes(Paths.get(jsonFile));
        return parseFrom(content);
    }

    public static Schema parseFrom(byte[] content) throws UserException {
        Map<String, AttrDef> nameToAttrDef = new HashMap<>();
        Map<Integer, AttrDef> idToAttrDef = new HashMap<>();
        Map<String, TypeDef> nameToTypeDef = new HashMap<>();
        Map<Integer, TypeDef> idToTypeDef = new HashMap<>();
        Map<String, EdgeRelation> edgeNameToRelation = new HashMap<>();
        Map<Integer, EdgeRelation> edgeIdToRelation = new HashMap<>();

        JSONObject obj = new JSONObject(new String(content));

        JSONArray jsonAttrs = obj.getJSONArray("attr_defs");
        for (int i = 0; i < jsonAttrs.length(); ++i) {
            JSONObject jsonAttr = jsonAttrs.getJSONObject(i);
            int attrTypeId = jsonAttr.getInt("type");
            String attrName = jsonAttr.getString("name");
            DataType dataType = DataType.parseString(jsonAttr.getString("value_type"));
            AttrDef attrDef = new AttrDef(attrTypeId, attrName, dataType);
            nameToAttrDef.put(attrName, attrDef);
            idToAttrDef.put(attrTypeId, attrDef);
        }

        JSONArray jsonVertices = obj.getJSONArray("vertex_defs");
        for (int i = 0; i < jsonVertices.length(); ++i) {
            JSONObject jsonVertex = jsonVertices.getJSONObject(i);
            JSONArray jsonAttrTypes = jsonVertex.getJSONArray("attr_types");
            List<AttrDef> vertexAttrs = new ArrayList<>();
            for (int j = 0; j < jsonAttrTypes.length(); ++j) {
                vertexAttrs.add(idToAttrDef.get(jsonAttrTypes.getInt(j)));
            }
            int vertexTypeId = jsonVertex.getInt("vtype");
            String vertexName = jsonVertex.getString("name");
            TypeDef vertexTypeDef = new TypeDef(TypeKind.VERTEX, vertexTypeId, vertexName, vertexAttrs);
            nameToTypeDef.put(vertexName, vertexTypeDef);
            idToTypeDef.put(vertexTypeId, vertexTypeDef);
        }

        JSONArray jsonEdges = obj.getJSONArray("edge_defs");
        for (int i = 0; i < jsonEdges.length(); ++i) {
            JSONObject jsonEdge = jsonEdges.getJSONObject(i);
            JSONArray jsonAttrTypes = jsonEdge.getJSONArray("attr_types");
            List<AttrDef> edgeAttrs = new ArrayList<>();
            for (int j = 0; j < jsonAttrTypes.length(); ++j) {
                edgeAttrs.add(idToAttrDef.get(jsonAttrTypes.getInt(j)));
            }
            int edgeTypeId = jsonEdge.getInt("etype");
            String edgeName = jsonEdge.getString("name");
            TypeDef edgeTypeDef = new TypeDef(TypeKind.EDGE, edgeTypeId, edgeName, edgeAttrs);
            nameToTypeDef.put(edgeName, edgeTypeDef);
            idToTypeDef.put(edgeTypeId, edgeTypeDef);
        }

        JSONArray jsonRelations = obj.getJSONArray("edge_relation_defs");
        for (int i = 0; i < jsonRelations.length(); ++i) {
            JSONObject jsonRelation = jsonRelations.getJSONObject(i);
            int edgeTypeId = jsonRelation.getInt("etype");
            String edgeName = idToTypeDef.get(edgeTypeId).getName();
            int srcVertexTypeId = jsonRelation.getInt("src_vtype");
            String srcVertexName = idToTypeDef.get(srcVertexTypeId).getName();
            int dstVertexTypeId = jsonRelation.getInt("dst_vtype");
            String dstVertexName = idToTypeDef.get(dstVertexTypeId).getName();
            EdgeRelation edgeRelation = new EdgeRelation(
                    edgeTypeId, edgeName, srcVertexTypeId, srcVertexName, dstVertexTypeId, dstVertexName);
            edgeNameToRelation.put(edgeName, edgeRelation);
            edgeIdToRelation.put(edgeTypeId, edgeRelation);
        }

        return new Schema(nameToAttrDef, idToAttrDef, nameToTypeDef, idToTypeDef, edgeNameToRelation, edgeIdToRelation);
    }

    public void dumpTo(String file) throws IOException {
        JSONArray jsonAttrs = new JSONArray();
        for (AttrDef attrDef : this.getAllAttrDefinitions()) {
            JSONObject jsonAttr = new JSONObject();
            jsonAttr.put("type", attrDef.getTypeId());
            jsonAttr.put("name", attrDef.getName());
            jsonAttr.put("value_type", attrDef.getDataType().toString());
            jsonAttrs.put(jsonAttr);
        }

        JSONArray jsonVertices = new JSONArray();
        JSONArray jsonEdges = new JSONArray();
        for (TypeDef typeDef : this.getAllTypeDefinitions()) {
            JSONArray jsonAttrTypes = new JSONArray();
            for (AttrDef attrDef : typeDef.getAttributes()) {
                jsonAttrTypes.put(attrDef.getTypeId());
            }
            if (typeDef.getTypeKind() == TypeKind.VERTEX) {
                JSONObject jsonVertex = new JSONObject();
                jsonVertex.put("vtype", typeDef.getTypeId());
                jsonVertex.put("name", typeDef.getName());
                jsonVertex.put("attr_types", jsonAttrTypes);
                jsonVertices.put(jsonVertex);
            } else {
                JSONObject jsonEdge = new JSONObject();
                jsonEdge.put("etype", typeDef.getTypeId());
                jsonEdge.put("name", typeDef.getName());
                jsonEdge.put("attr_types", jsonAttrTypes);
                jsonEdges.put(jsonEdge);
            }
        }

        JSONArray jsonRelations = new JSONArray();
        for (EdgeRelation edgeRelation : this.getAllEdgeRelations()) {
            JSONObject jsonRelation = new JSONObject();
            jsonRelation.put("etype", edgeRelation.getEdgeTypeId());
            jsonRelation.put("src_vtype", edgeRelation.getSrcTypeId());
            jsonRelation.put("dst_vtype", edgeRelation.getDstTypeId());
            jsonRelations.put(jsonRelation);
        }

        JSONObject obj = new JSONObject();
        obj.put("attr_defs", jsonAttrs);
        obj.put("vertex_defs", jsonVertices);
        obj.put("edge_defs", jsonEdges);
        obj.put("edge_relation_defs", jsonRelations);

        Files.write(Paths.get(file), obj.toString().getBytes());
    }

    private Schema(Map<String, AttrDef> nameToAttrDef,
                   Map<Integer, AttrDef> idToAttrDef,
                   Map<String, TypeDef> nameToTypeDef,
                   Map<Integer, TypeDef> idToTypeDef,
                   Map<String, EdgeRelation> edgeNameToRelation,
                   Map<Integer, EdgeRelation> edgeIdToRelation) {
        this.nameToAttrDef = new HashMap<>(nameToAttrDef);
        this.idToAttrDef = new HashMap<>(idToAttrDef);
        this.nameToTypeDef = new HashMap<>(nameToTypeDef);
        this.idToTypeDef = new HashMap<>(idToTypeDef);
        this.edgeNameToRelation = new HashMap<>(edgeNameToRelation);
        this.edgeIdToRelation = new HashMap<>(edgeIdToRelation);
    }

    public static Builder newBuilder() {
        return new Builder();
    }

    public static Builder newBuilder(Schema schema) {
        return new Builder(schema);
    }

    public static class Builder {
        private Map<String, DataType> attrToDataType;
        private Map<String, Set<String>> vertexToAttr;
        private Map<String, Set<String>> edgeToAttr;
        private Map<String, RawEdgeRelation> edgeToRawRelation;

        private Builder() {
            this.attrToDataType = new HashMap<>();
            this.vertexToAttr = new HashMap<>();
            this.edgeToAttr = new HashMap<>();
            this.edgeToRawRelation = new HashMap<>();
        }

        private Builder(Schema schema) {
            this.attrToDataType = new HashMap<>();
            this.vertexToAttr = new HashMap<>();
            this.edgeToAttr = new HashMap<>();
            this.edgeToRawRelation = new HashMap<>();
            for (AttrDef attrDef : schema.getAllAttrDefinitions()) {
                this.attrToDataType.put(attrDef.getName(), attrDef.getDataType());
            }
            for (TypeDef typeDef : schema.getAllTypeDefinitions()) {
                Set<String> attrNames =
                        typeDef.getAttributes().stream().map(AttrDef::getName).collect(Collectors.toSet());
                if (typeDef.getTypeKind() == TypeKind.VERTEX) {
                    this.vertexToAttr.put(typeDef.getName(), attrNames);
                } else {
                    this.edgeToAttr.put(typeDef.getName(), attrNames);
                }
            }
            for (EdgeRelation edgeRelation : schema.getAllEdgeRelations()) {
                this.edgeToRawRelation.put(edgeRelation.getEdgeName(), new RawEdgeRelation(
                        edgeRelation.getEdgeName(), edgeRelation.getSrcName(), edgeRelation.getDstName()));
            }
        }

        public Builder addAttribute(String attrName, DataType dataType) {
            this.attrToDataType.put(attrName, dataType);
            return this;
        }

        public Builder removeAttribute(String attrName) {
            this.attrToDataType.remove(attrName);
            return this;
        }

        public Builder addVertex(String vertexName, List<String> vertexAttrNames) {
            this.vertexToAttr.put(vertexName, new HashSet<>(vertexAttrNames));
            return this;
        }

        public Builder removeVertex(String vertexName) {
            this.vertexToAttr.remove(vertexName);
            return this;
        }

        public Builder addVertexAttribute(String vertexName, String attrName) {
            this.vertexToAttr.get(vertexName).add(attrName);
            return this;
        }

        public Builder removeVertexAttribute(String vertexName, String attrName) {
            this.vertexToAttr.get(vertexName).remove(attrName);
            return this;
        }

        public Builder addEdge(String edgeName, String srcName, String dstName, List<String> edgeAttrNames) {
            this.edgeToAttr.put(edgeName, new HashSet<>(edgeAttrNames));
            this.edgeToRawRelation.put(edgeName, new RawEdgeRelation(edgeName, srcName, dstName));
            return this;
        }

        public Builder removeEdge(String edgeName) {
            this.edgeToAttr.remove(edgeName);
            this.edgeToRawRelation.remove(edgeName);
            return this;
        }

        public Builder addEdgeAttribute(String edgeName, String attrName) {
            this.edgeToAttr.get(edgeName).add(attrName);
            return this;
        }

        public Builder removeEdgeAttribute(String edgeName, String attrName) {
            this.edgeToAttr.get(edgeName).remove(attrName);
            return this;
        }

        public Schema build() {
            Map<String, AttrDef> nameToAttrDef = new HashMap<>();
            Map<Integer, AttrDef> idToAttrDef = new HashMap<>();
            Map<String, TypeDef> nameToTypeDef = new HashMap<>();
            Map<Integer, TypeDef> idToTypeDef = new HashMap<>();
            Map<String, EdgeRelation> edgeNameToRelation = new HashMap<>();
            Map<Integer, EdgeRelation> edgeIdToRelation = new HashMap<>();

            int attrTypeId = 0;
            for (Map.Entry<String, DataType> entry : this.attrToDataType.entrySet()) {
                AttrDef attrDef = new AttrDef(attrTypeId, entry.getKey(), entry.getValue());
                nameToAttrDef.put(entry.getKey(), attrDef);
                idToAttrDef.put(attrTypeId, attrDef);
                attrTypeId++;
            }

            int typeId = 0;
            for (Map.Entry<String, Set<String>> entry : this.vertexToAttr.entrySet()) {
                String vertexName = entry.getKey();
                List<AttrDef> attrList = new ArrayList<>();
                for (String attrName : entry.getValue()) {
                    if (nameToAttrDef.containsKey(attrName)) {
                        attrList.add(nameToAttrDef.get(attrName));
                    }
                }
                TypeDef typeDef = new TypeDef(TypeKind.VERTEX, typeId, vertexName, attrList);
                nameToTypeDef.put(vertexName, typeDef);
                idToTypeDef.put(typeId, typeDef);
                typeId++;
            }
            for (Map.Entry<String, Set<String>> entry : this.edgeToAttr.entrySet()) {
                String edgeName = entry.getKey();
                List<AttrDef> attrList = new ArrayList<>();
                for (String attrName : entry.getValue()) {
                    if (nameToAttrDef.containsKey(attrName)) {
                        attrList.add(nameToAttrDef.get(attrName));
                    }
                }
                TypeDef typeDef = new TypeDef(TypeKind.EDGE, typeId, edgeName, attrList);
                nameToTypeDef.put(edgeName, typeDef);
                idToTypeDef.put(typeId, typeDef);
                typeId++;
            }

            for (RawEdgeRelation rawRelation : this.edgeToRawRelation.values()) {
                String edgeName = rawRelation.getEdgeName();
                if (!nameToTypeDef.containsKey(edgeName)) {
                    continue;
                }
                int edgeTid = nameToTypeDef.get(edgeName).getTypeId();

                String srcName = rawRelation.getSrcName();
                if (!nameToTypeDef.containsKey(srcName)) {
                    continue;
                }
                int srcTid = nameToTypeDef.get(srcName).getTypeId();

                String dstName = rawRelation.getDstName();
                if (!nameToTypeDef.containsKey(dstName)) {
                    continue;
                }
                int dstTid = nameToTypeDef.get(dstName).getTypeId();

                EdgeRelation edgeRelation = new EdgeRelation(edgeTid, edgeName, srcTid, srcName, dstTid, dstName);
                edgeNameToRelation.put(edgeName, edgeRelation);
                edgeIdToRelation.put(edgeTid, edgeRelation);
            }

            return new Schema(nameToAttrDef, idToAttrDef, nameToTypeDef, idToTypeDef, edgeNameToRelation, edgeIdToRelation);
        }
    }
}
