package org.aliyun.gsl_client;

import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.Context;
import org.aliyun.gsl_client.parser.Plan;
import org.aliyun.gsl_client.parser.PlanNode;
import org.aliyun.gsl_client.parser.optimizer.FusionRule;
import org.aliyun.gsl_client.parser.optimizer.Rule;
import org.aliyun.gsl_client.status.ErrorCode;

public class Traversal {
  // private fields.
  public Context context;
  public PlanNode plan_node;

  public Traversal() {
  }

  public Traversal(PlanNode node, Context context) {
    this.plan_node = node;
    this.context = context;
  }

  public Traversal feed(DataSource source) {
    this.plan_node.setSource(source);
    return this;
  }

  // Traverse to the Vertex along with the given path, which must be
  // the out edge type of the upstream type of vertex.
  public Traversal outV(String path) throws UserException {
    // 1. new PlanNode add to plan
    Plan plan = context.getPlan();
    try {
      PlanNode node = plan.addEdgeSamplerNode(context.srcType(path), context.type(path));
      node.addPropertyFilter();
      this.plan_node.addChild(context.getKind(), 0, node);

      // Traverse to the downstream
      this.plan_node = node;
      this.context.setType(context.dstType(path));
      this.context.setKind(1);
      return this;
    } catch (Exception e) {
      e.printStackTrace();
      throw new UserException(ErrorCode.PAESE_ERROR, "Parse Query Failed.");
    }
  }

  // Traverse to the Edge along with the given path, which must be
  // the out edge type of the upstream type of vertex.
  public Traversal outE(String path) throws UserException {
    Plan plan = context.getPlan();
    try {
      PlanNode node = plan.addEdgeSamplerNode(context.srcType(path), context.type(path));

      this.plan_node.addChild(0, 0, node);

      this.plan_node = node;
      return this;
    } catch (Exception e) {
      e.printStackTrace();
      throw new UserException(ErrorCode.PAESE_ERROR, "Parse Query Failed.");
    }
  }

  public Traversal inV() throws UserException {
    // TODO
    return this;
  }

  public Traversal outV() throws UserException {
    // TODO
    return this;
  }

  // sample neighbor counts for each input node
  public Traversal sample(int fanout) throws UserException {
    this.plan_node.addParam("fanout", fanout);
    return this;
  }

  // strategy for sampler.
  public Traversal by(String strategy) throws UserException {
    // TODO: Map string strategy to int.
    this.plan_node.addParam("strategy", 0);
    return this;
  }

  // Set property version for Vertex
  public Traversal properties(Integer version, String... keys) throws UserException {
    Plan plan = context.getPlan();
    try {
      PlanNode node = plan.addVertexSamplerNode(context.getType());
      node.addParam("versions", version);
      this.plan_node.addChild(context.getKind(), 0, node);
      // Add PropertyFilter for the PlanNode.
      // No Traverse action happened.
      return this;
    } catch (Exception e) {
      e.printStackTrace();
      throw new UserException(ErrorCode.PAESE_ERROR, "Parse Query Failed.");
    }
  }

  public Traversal alias(String name) {
    this.plan_node.setAlias(name);
    return this;
  }

  public Query values() throws UserException {
    Plan plan = context.getPlan();
    try {
      plan.finalize();
      Rule rule = new FusionRule();
      plan.optimize(rule);
    } catch (Exception e) {
      e.printStackTrace();
      throw new UserException(ErrorCode.PAESE_ERROR, "Query values() Failed.");
    }
    return new Query(plan);
  }
}
