package org.aliyun;

import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.schema.*;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;

public class SchemaTest extends TestCase {
  public SchemaTest(String testName) {
    super(testName);
  }

  public static Test suite() {
    return new TestSuite(SchemaTest.class);
  }

  public void testSchemaParsing() throws UserException, IOException {
    Schema schema = Schema.parseFrom("../conf/schema.template.json");

    assertEquals(schema.getAttrDef("timestamp").getDataType(), DataType.INT64);
    assertEquals(schema.getAttrDef("weight").getTypeId(), 1);
    assertEquals(schema.getAttrDef(2).getName(), "label");
    assertEquals(schema.getAttrDef("attr1").getTypeId(), 3);
    assertEquals(schema.getAttrDef(4).getDataType(), DataType.FLOAT64);

    TypeDef userDef = schema.getTypeDef("user");
    assertEquals(userDef.getTypeKind(), TypeKind.VERTEX);
    assertEquals(userDef.getTypeId(), 0);
    assertEquals(userDef.getAttributes().get(0).getName(), "timestamp");

    TypeDef itemDef = schema.getTypeDef(1);
    assertEquals(itemDef.getTypeKind(), TypeKind.VERTEX);
    assertEquals(itemDef.getName(), "item");
    assertEquals(itemDef.getAttributes().size(), 4);

    assertEquals(schema.getTypeDef("click").getTypeId(), 2);
    assertEquals(schema.getTypeDef(3).getTypeKind(), TypeKind.EDGE);
    assertEquals(schema.getTypeDef("knows").getAttributes().get(1).getName(), "weight");

    assertEquals(schema.getRelation("click").getSrcTypeId(), 0);
    assertEquals(schema.getRelation("buy").getDstTypeId(), 1);
    assertEquals(schema.getRelation(4).getSrcName(), "user");
  }

  public void testSchemaBuilding() throws IOException {
    Schema.Builder builder = Schema.newBuilder();
    Schema schema = builder.addAttribute("name", DataType.STRING)
            .addAttribute("age", DataType.parseString("int32"))
            .addAttribute("country", DataType.STRING)
            .addAttribute("content", DataType.parseString("String"))
            .addAttribute("pagerank", DataType.FLOAT64)
            .addAttribute("time", DataType.parseString("STRING"))
            .addVertex("person", Arrays.asList("name", "age", "country"))
            .addVertex("post", Arrays.asList("content", "pagerank"))
            .addVertex("comment", Arrays.asList("content", "pagerank"))
            .addEdge("likes", "person", "post", Arrays.asList("time", "pagerank"))
            .addEdge("creates", "person", "comment", Collections.singletonList("time"))
            .build();

    schema.dumpTo("schema_ut.json");
    Files.delete(Paths.get("schema_ut.json"));

    Schema.Builder builder2 = Schema.newBuilder(schema);
    Schema schema2 = builder2.removeAttribute("country")
            .removeAttribute("pagerank")
            .addAttribute("city", DataType.STRING)
            .addAttribute("pagerank2", DataType.FLOAT64)
            .removeVertexAttribute("person", "country")
            .addVertexAttribute("person", "city")
            .addVertexAttribute("post", "pagerank2")
            .removeVertex("comment")
            .addVertex("message", Arrays.asList("content", "pagerank2"))
            .removeEdgeAttribute("likes", "pagerank")
            .addEdgeAttribute("likes", "pagerank2")
            .removeEdge("creates")
            .addEdge("follows", "person", "person", Arrays.asList("time", "pagerank2"))
            .build();

    schema2.dumpTo("schema2_ut.json");
    Files.delete(Paths.get("schema2_ut.json"));
  }
}
