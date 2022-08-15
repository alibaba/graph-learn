package org.aliyun.gsl_client.parser;

import java.util.HashMap;
import java.util.Vector;

import org.aliyun.gsl_client.DataSource;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.Map;

/**
 * Convert the GSL api to Logical Query Plan.
 */
public abstract class PlanNode {
  protected int id;
  protected String alias = "";
  protected Map<String, Integer> params = new HashMap<>();
  protected Vector<Link> childLinks = new Vector<>();
  protected boolean withProperty = true;
  protected DataSource source;

  public PlanNode() {}

  public PlanNode(int id, Map<String, Integer> params, Vector<Link> childLinks) {
    this.id = id;
    this.params = params;
    this.childLinks = childLinks;
  }

  public int getId() {
    return this.id;
  }

  public void setId(int id) {
    this.id = id;
  }


  public String getAlias() {
    return this.alias;
  }

  public void setAlias(String alias) {
    this.alias = alias;
  }

  public Integer getParam(String key) {
    return params.get(key);
  }

  public Vector<Link> getChildLinks() {
    return this.childLinks;
  }

  public void addChild(Integer srcOutput, Integer dstInput, PlanNode node) {
    Link link = new Link(srcOutput, dstInput, node);
    childLinks.add(link);
  }

  public void addParam(String key, Integer value) {
    params.put(key, value);
  }

  public void addPropertyFilter() {
    withProperty = false;
  }

  public String getKind() {
    return "UNSPECIFIED";
  }

  protected String getType() {
    return "UNSPECIFIED";
  }

  public DataSource getSource() {
    return source;
  }

  public void setSource(DataSource source) {
    this.source = source;
  }

  public boolean checkValid() {
    return true;
  }

  public void copyFrom(PlanNode node) {
    id = node.id;
    params = node.params;
    childLinks = node.childLinks;
    withProperty = node.withProperty;
  }

  public JSONObject toJson() {
    JSONObject jsonObj = new JSONObject();
    jsonObj.put("id", id)
           .put("kind", getKind())
           .put("type", getType())
           .put("links", linkArray())
           .put("params", paramsArray())
           .put("filter", filterObj());
    return jsonObj;
  }

  private JSONArray linkArray() {
    JSONArray array = new JSONArray();
    childLinks.forEach((link) -> {
      JSONObject linkObj = new JSONObject();
      linkObj.put("node", link.node.getId())
             .put("src_output", link.srcOutput)
             .put("dst_input", link.dstInput);
      array.put(linkObj);
    });
    return array;
  }

  private JSONArray paramsArray() {
    JSONArray array = new JSONArray();
    params.entrySet().forEach((param) -> {
      JSONObject paramObj = new JSONObject();
      paramObj.put("key", "\\" + param.getKey() + "\\")
              .put("value", param.getValue());
      array.put(paramObj);
    });
    return array;
  }

  private JSONObject filterObj() {
    JSONObject obj = new JSONObject();
    obj.put("weighted", withProperty)
       .put("labeled", withProperty)
       .put("attributed", withProperty);
    return obj;
  }
}

class RootNode extends PlanNode {
  public RootNode() {
    this.id = 0;
  }

  public RootNode(int id,
                  Map<String, Integer> params,
                  Vector<Link> childLinks) {
    this.id = id;
    this.params = params;
    this.childLinks = childLinks;
  }

  @Override
  public String getKind() {
    return "SOURCE";
  }

  @Override
  protected String getType() {
    return "VERTEX";
  }

  @Override
  public boolean checkValid() {
    if (params.containsKey("vtype")) {
      return true;
    }
    return false;
  }
}

class VertexSamplerNode extends PlanNode {
  public VertexSamplerNode(int opid) {
    this.id = opid;
  }

  public VertexSamplerNode(int id,
                           Map<String, Integer> params,
                           Vector<Link> childLinks) {
    this.id = id;
    this.params = params;
    this.childLinks = childLinks;
  }

  @Override
  public String getKind() {
    return "VERTEX_SAMPLER";
  }

  @Override
  protected String getType() {
    return "VERTEX";
  }

  @Override
  public boolean checkValid() {
    if (params.containsKey("vtype") && params.containsKey("versions")) {
      return true;
    }
    return false;
  }
}

class EdgeSamplerNode extends PlanNode {
  public EdgeSamplerNode(int opid) {
    this.id = opid;
  }

  public EdgeSamplerNode(int id,
                         Map<String, Integer> params,
                         Vector<Link> childLinks) {
    this.id = id;
    this.params = params;
    this.childLinks = childLinks;
  }

  @Override
  public String getKind() {
    return "EDGE_SAMPLER";
  }

  @Override
  protected String getType() {
    return "EDGE";
  }

  @Override
  public boolean checkValid() {
    if (params.containsKey("vtype")
        && params.containsKey("etype")
        && params.containsKey("fanout")
        && params.containsKey("strategy")) {
      return true;
    }
    return false;
  }
}