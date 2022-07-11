package org.aliyun.gsl_client;

import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.Plan;
import org.aliyun.gsl_client.parser.PlanNode;

public class Query {
  private Plan plan;
  private String id;

  public Query(Plan plan) {
    this.plan = plan;
    this.id = "-1";
  }

  public Plan getPlan() {
    return this.plan;
  }

  public void setPlan(Plan plan) {
    this.plan = plan;
  }

  public String getId() {
    return this.id;
  }

  public void setId(String id) {
    this.id = id;
  }

  public void feed(String alias, DataSource source) throws UserException {
    PlanNode node = plan.getNode(alias);
    node.setSource(source);
  }

}
