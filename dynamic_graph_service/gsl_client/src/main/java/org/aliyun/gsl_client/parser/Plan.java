package org.aliyun.gsl_client.parser;

import java.util.ArrayList;
import java.util.Vector;
import java.util.concurrent.atomic.AtomicInteger;

import org.aliyun.dgs.PlanNode.ChildLink;
import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.optimizer.Rule;
import org.aliyun.gsl_client.status.ErrorCode;
import org.json.JSONArray;
import org.json.JSONObject;

public class Plan {
  private Vector<PlanNode> planNodes = new Vector<>();
  private AtomicInteger size = new AtomicInteger(0);
  private boolean finailzed = false;

  public PlanNode addRoot(int vtype) throws UserException {
    if (finailzed) {
      throw new UserException(ErrorCode.PAESE_ERROR, "Query has finalized.");
    }
    PlanNode node = new RootNode();
    node.addParam("vtype", vtype);
    this.planNodes.add(node);
    this.size.incrementAndGet();
    return node;
  }

  public PlanNode addEdgeSamplerNode(int vtype, int etype) throws UserException {
    if (finailzed) {
      throw new UserException(ErrorCode.PAESE_ERROR, "Query has finalized.");
    }
    PlanNode node = new EdgeSamplerNode(this.size.getAndIncrement());
    node.addParam("vtype", vtype);
    node.addParam("etype", etype);
    this.planNodes.add(node);
    return node;
  }

  public PlanNode addVertexSamplerNode(int vtype) throws UserException {
    if (finailzed) {
      throw new UserException(ErrorCode.PAESE_ERROR, "Query has finalized.");
    }
    PlanNode node = new VertexSamplerNode(this.size.getAndIncrement());
    node.addParam("vtype", vtype);
    this.planNodes.add(node);
    return node;
  }

  public void finalize() throws UserException {
    if (finailzed) {
      throw new UserException(ErrorCode.PAESE_ERROR, "Query has finalized.");
    }
    this.finailzed = true;
  }

  public PlanNode root() throws UserException {
    if (!finailzed) {
      throw new UserException(ErrorCode.PAESE_ERROR, "Query should been finalized.");
    }
    if (size() < 1) {
      throw new UserException(ErrorCode.PAESE_ERROR, "Query is empty.");
    }
    return planNodes.get(0);
  }

  public PlanNode getNode(String alias) throws UserException {
    if (finailzed) {
      throw new UserException(ErrorCode.PAESE_ERROR, "Query has finalized.");
    }
    for (int i = 0; i < size(); ++i) {
      PlanNode node = planNodes.get(i);
      if (node.getAlias().equals(alias)) {
        return node;
      }
    }
    throw new UserException(ErrorCode.PAESE_ERROR, "Query has no node alias as " + alias);
  }

  public ArrayList<PlanNode> getEgoGraphNodes(String alias) throws UserException {
    ArrayList<PlanNode> nodes = new ArrayList<>();
    for (int i = 0; i < size(); ++i) {
      PlanNode root = planNodes.get(i);
      if (root.getAlias().equals(alias)) {
        nodes.add(root);
        while (root.getChildLinks().size() > 0) {
          // For ego graph, only one child supported.
          root = root.getChildLinks().get(0).node;
          nodes.add(root);
        }
      }
    }
    return nodes;
  }

  public Vector<PlanNode> nodes() {
    return planNodes;
  }

  public int size() {
    return this.size.get();
  }

  public void optimize(Rule... rules) throws UserException {
    if (!finailzed) {
      throw new UserException(ErrorCode.PAESE_ERROR, "Fuse plan without finalized.");
    }
    for (Rule rule : rules) {
      rule.Act(planNodes, size);
    }
  }

  public JSONObject toJson() throws UserException {
    if (!finailzed) {
      throw new UserException(ErrorCode.PAESE_ERROR, "Query should be finalized.");
    }
    JSONObject obj = new JSONObject();
    JSONArray nodesArr = new JSONArray();
    for (int idx = 0; idx < size.get(); ++idx) {
      nodesArr.put(planNodes.get(idx).toJson());
    }
    JSONObject plan_obj = new JSONObject();
    plan_obj.put("plan_nodes", nodesArr);

    obj.put("priority", 0)
       .put("query_plan", plan_obj)
       .put("query_id", 0);
    return obj;
  }
}
