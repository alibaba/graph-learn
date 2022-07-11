package org.aliyun.gsl_client.parser.optimizer;

import java.util.Vector;
import java.util.concurrent.atomic.AtomicInteger;

import org.aliyun.gsl_client.parser.PlanNode;

public abstract class Rule {
  public abstract void Act(Vector<PlanNode> nodes, AtomicInteger size);

}
