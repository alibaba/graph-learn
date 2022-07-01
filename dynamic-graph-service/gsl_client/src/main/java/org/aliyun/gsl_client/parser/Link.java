package org.aliyun.gsl_client.parser;

public class Link {
  public Integer srcOutput;
  public Integer dstInput;
  public PlanNode node;

  public Link(Integer srcOutput, Integer dstInput, PlanNode node) {
    this.srcOutput = srcOutput;
    this.dstInput = dstInput;
    this.node = node;
  }
}
