package org.aliyun.gsl_client.parser;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
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
      throw new UserException(ErrorCode.PARSE_ERROR, "Query has finalized.");
    }
    PlanNode node = new RootNode();
    node.addParam("vtype", vtype);
    this.planNodes.add(node);
    this.size.incrementAndGet();
    return node;
  }

  public void addNode(PlanNode node) throws UserException {
    if (finailzed) {
      throw new UserException(ErrorCode.PARSE_ERROR, "Query has finalized.");
    }
    this.planNodes.add(node);
    this.size.incrementAndGet();
  }

  public void addNode(int idx, PlanNode node) throws UserException {
    if (finailzed) {
      throw new UserException(ErrorCode.PARSE_ERROR, "Query has finalized.");
    }
    if (this.planNodes.size() <= idx) {
      this.planNodes.setSize(idx + 1);
    }
    this.planNodes.set(idx, node);
  }

  public PlanNode addEdgeSamplerNode(int vtype, int etype) throws UserException {
    if (finailzed) {
      throw new UserException(ErrorCode.PARSE_ERROR, "Query has finalized.");
    }
    PlanNode node = new EdgeSamplerNode(this.size.getAndIncrement());
    node.addParam("vtype", vtype);
    node.addParam("etype", etype);
    this.planNodes.add(node);
    return node;
  }

  public PlanNode addVertexSamplerNode(int vtype) throws UserException {
    if (finailzed) {
      throw new UserException(ErrorCode.PARSE_ERROR, "Query has finalized.");
    }
    PlanNode node = new VertexSamplerNode(this.size.getAndIncrement());
    node.addParam("vtype", vtype);
    this.planNodes.add(node);
    return node;
  }

  public void finalize() throws UserException {
    if (finailzed) {
      throw new UserException(ErrorCode.PARSE_ERROR, "Query has finalized.");
    }
    this.finailzed = true;
  }

  public PlanNode root() throws UserException {
    if (!finailzed) {
      throw new UserException(ErrorCode.PARSE_ERROR, "Query should been finalized.");
    }
    if (size() < 1) {
      throw new UserException(ErrorCode.PARSE_ERROR, "Query is empty.");
    }
    return planNodes.get(0);
  }

  public PlanNode getNode(String alias) throws UserException {
    if (finailzed) {
      throw new UserException(ErrorCode.PARSE_ERROR, "Query has finalized.");
    }
    for (int i = 0; i < size(); ++i) {
      PlanNode node = planNodes.get(i);
      if (node.getAlias().equals(alias)) {
        return node;
      }
    }
    throw new UserException(ErrorCode.PARSE_ERROR, "Query has no node alias as " + alias);
  }

  public ArrayList<PlanNode> getEgoGraphNodes() throws UserException {
    ArrayList<PlanNode> nodes = new ArrayList<>();
    Queue<PlanNode> curLayer = new LinkedList<PlanNode>();
    PlanNode root = planNodes.get(0);
    curLayer.add(root);
    while (!curLayer.isEmpty()) {
      PlanNode cur = curLayer.remove();
      nodes.add(cur);
      Vector<Link> links = cur.getChildLinks();
      for (int idx = 0; idx < links.size(); ++idx) {
        curLayer.add(links.get(idx).node);
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
      throw new UserException(ErrorCode.PARSE_ERROR, "Fuse plan without finalized.");
    }
    for (Rule rule : rules) {
      rule.Act(planNodes, size);
    }
  }

  public JSONObject toJson() throws UserException {
    if (!finailzed) {
      throw new UserException(ErrorCode.PARSE_ERROR, "Query should be finalized.");
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

  public static Plan parseFrom(String jsonFile) throws UserException, IOException {
    byte[] content = Files.readAllBytes(Paths.get(jsonFile));
    return parseFrom(content);
  }

  public static Plan parseFrom(byte[] content) throws UserException {
    JSONObject obj = new JSONObject(new String(content));
    Plan plan = new Plan();
    JSONObject planObj = obj.getJSONObject("query_plan");
    JSONArray jsonAttrs = planObj.getJSONArray("plan_nodes");
    for (int i = 0; i < jsonAttrs.length(); ++i) {
      JSONObject nodeobj = jsonAttrs.getJSONObject(i);
      int id = nodeobj.getInt("id");
      if (id == 0) { // Parse Node from root
        parseNode(nodeobj, jsonAttrs, plan);
      }
    }
    plan.size.set(jsonAttrs.length());
    plan.finalize();
    return plan;
  }

  private static PlanNode parseNode(JSONObject nodeObj, JSONArray nodes, Plan plan) throws UserException {
    int id = nodeObj.getInt("id");
    String kind = nodeObj.getString("kind");

    Vector<Link> childLinks = new Vector<>();
    JSONArray links = nodeObj.getJSONArray("links");
    for (int j = 0; j < links.length(); ++j) {
      JSONObject childLink = links.getJSONObject(j);
      int nid = childLink.getInt("node");
      PlanNode node = parseNode(nodes.getJSONObject(nid), nodes, plan);
      int srcOutput = childLink.getInt("src_output");
      int dstInput = childLink.getInt("dst_input");
      Link link = new Link(srcOutput, dstInput, node);
      childLinks.add(link);
    }

    Map<String, Integer> params = new HashMap<>();
    JSONArray paramsArray = nodeObj.getJSONArray("params");
    for (int j = 0; j < paramsArray.length(); ++j) {
      JSONObject paramObj = paramsArray.getJSONObject(j);
      String key = paramObj.getString("key");
      int value = paramObj.getInt("value");
      params.put(key, value);
    }
    switch (kind) {
      case "SOURCE":
        RootNode node = new RootNode(id, params, childLinks);
        plan.addNode(id, node);
        return node;
      case "VERTEX_SAMPLER":
        VertexSamplerNode vnode = new VertexSamplerNode(id, params, childLinks);
        plan.addNode(id, vnode);
        return vnode;
      case "EDGE_SAMPLER":
        EdgeSamplerNode enode = new EdgeSamplerNode(id, params, childLinks);
        plan.addNode(id, enode);
        return enode;
      default:
        throw new UserException(ErrorCode.PARSE_ERROR, "Parse error with planNode kind=" + kind);
    }
  }
}
