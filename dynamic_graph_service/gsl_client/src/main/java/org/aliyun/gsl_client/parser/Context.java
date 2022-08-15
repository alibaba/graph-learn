package org.aliyun.gsl_client.parser;

import org.aliyun.gsl_client.Graph;
import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.schema.Schema;

public class Context {
  private Plan plan;
  private Graph graph;
  private int kind; // 0: Traversal to Vertex, 1: Traversal to Edge
  private int typeId;
  private Schema schema;

  public Context(Plan plan, Graph graph, int kind,
                 String typeName, Schema schema) throws UserException {
    this.plan = plan;
    this.graph = graph;
    this.kind = kind;
    this.schema = schema;
    this.typeId = schema.getTypeDef(typeName).getTypeId();
  }

  public Plan getPlan() {
    return this.plan;
  }

  public void setPlan(Plan plan) {
    this.plan = plan;
  }

  public Graph getGraph() {
    return this.graph;
  }

  public void setGraph(Graph graph) {
    this.graph = graph;
  }

  public int getKind() {
    return this.kind;
  }

  public void setKind(int kind) {
    this.kind = kind;
  }

  public int getType() {
    return this.typeId;
  }

  public void setType(int typeId) {
    this.typeId = typeId;
  }

  public int type(String typeName) throws UserException {
    return schema.getTypeDef(typeName).getTypeId();
  }

  public int srcType(String typeName) throws UserException {
    return schema.getRelation(typeName).getSrcTypeId();
  }

  public int dstType(String typeName) throws UserException {
    return schema.getRelation(typeName).getDstTypeId();
  }
}
