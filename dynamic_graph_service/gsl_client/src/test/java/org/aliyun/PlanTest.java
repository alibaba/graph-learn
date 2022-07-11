package org.aliyun;

import org.aliyun.gsl_client.Graph;
import org.aliyun.gsl_client.Query;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/**
 * Run these test cases after `python3 mock_server.py`.
 */
public class PlanTest extends TestCase {
  public PlanTest(String testName) {
    super(testName);
  }

  public static Test suite() {
    return new TestSuite(PlanTest.class);
  }

  public void testPlan() {
    String mock_server = "http://localhost:8088";

    System.out.println("Start to connect...");
    Graph g = Graph.connect(mock_server);
    System.out.println("Connected...");
    try {
      Query query = g.V("user").properties(10)
          .outV("knows").sample(15).by("topk").properties(10)
          .outV("knows").sample(10).by("topk").properties(10)
          .values();

      String plan_str = query.getPlan().toJson().toString();
      System.out.println(plan_str);

    } catch (Exception e) {
      System.out.println(e.getMessage());
    }
  }
}
