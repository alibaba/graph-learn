package org.aliyun.gsl_client;

import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.Context;
import org.aliyun.gsl_client.parser.Plan;
import org.aliyun.gsl_client.parser.PlanNode;
import org.aliyun.gsl_client.parser.optimizer.FusionRule;
import org.aliyun.gsl_client.parser.optimizer.Rule;
import org.aliyun.gsl_client.status.ErrorCode;

/**
 * Traversal is a walker on vertices and edges along with connected
 * paths. Traversal supports sampling, getting properties on one or
 * a batch of traversed vertices or edges.
 */
public class Traversal {
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

  /**
   * Traverse to the destination Vertex along with the given edge.
   * @param path, edge type begin current vertex type.
   * @return Traversal, which is on the destination vertex along with the path.
   * @throws UserException
   */
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

  /**
   * Traverse to the Edge along with the given path.
   * @param path, edge type begin current vertex type.
   * @return Traversal, which is on the Edge along with the path.
   * @throws UserException
   */
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

  /**
   * Not implemented yet.
   * @return
   * @throws UserException
   */
  public Traversal inV() throws UserException {
    return this;
  }

  /**
   * Not implemented yet.
   * @return
   * @throws UserException
   */
  public Traversal outV() throws UserException {
    return this;
  }

  /**
   * Sampling fanout for current traversed object.
   * @param fanout, neighbor count for each upstream vertex.
   * @return Traversal itself.
   * @throws UserException
   */
  public Traversal sample(int fanout) throws UserException {
    this.plan_node.addParam("fanout", fanout);
    return this;
  }

  /**
   * Sampling strategy for current traversed object.
   * @param strategy, sampling strategy.
   *    "topk_by_timestamp": sampling by topk timestamp.
   *    "edge_weight": sampling with distribution of edge weight.
   *    "random": random sampling.
   * @return Traversal itself.
   * @throws UserException
   */
  public Traversal by(String strategy) throws UserException {
    this.plan_node.addParam("strategy", 0);
    return this;
  }

  /**
   * Count of properties version for current traversed object.
   * @param version, count of version.
   * @param keys, filed filter for properties, not implemented yet.
   * @return Traversal, not move.
   * @throws UserException
   */
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

  /**
   * Add a name for each Traversal
   * @param name(String), each Traversal should be unique in one query.
   * @return Traversal itself.
   */
  public Traversal alias(String name) {
    this.plan_node.setAlias(name);
    return this;
  }

  /**
   * End up the query, and generate a optimized query plan.
   * @return Query
   * @throws UserException
   */
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
