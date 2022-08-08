package org.aliyun;

import org.aliyun.gsl_client.DataSource;
import org.aliyun.gsl_client.Graph;
import org.aliyun.gsl_client.Query;
import org.aliyun.gsl_client.Value;
import org.aliyun.gsl_client.ValueBuilder;
import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.Plan;
import org.aliyun.gsl_client.parser.PlanNode;
import org.aliyun.gsl_client.parser.schema.Schema;
import org.aliyun.gsl_client.predict.EgoGraph;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Vector;

class MockDataSource extends DataSource {
  Vector<Long> vids = new Vector<>();
  int cursor = 0;
  int capacity = 0;

  MockDataSource(int capacity) {
    this.capacity = capacity;
    for (int i = 0; i < capacity; ++i) {
      vids.add(Long.valueOf(i));
    }
  }

  public Long next() {
    Long vid = vids.get(cursor);
    cursor += 1;
    return vid;
  }

  public boolean hasNext() {
    return cursor < capacity;
  }

  public boolean seekTimestamp(Long timestamp) {
    return true;
  }
}

public class EgoGraphTest extends TestCase {
  public EgoGraphTest(String testName) {
    super(testName);
  }

  public static Test suite() {
    return new TestSuite(EgoGraphTest.class);
  }

  public void testGetEgoGraph() throws UserException, IOException {
    int iters = 5;
    DataSource source = new MockDataSource(iters);
    // Start coordinator first.

    Plan plan = Plan.parseFrom("../../conf/u2i/install_query.u2i.json");
    Query query = new Query(plan);

    query.feed(source);;
    Schema schema = Schema.parseFrom("../../conf/u2i/schema.u2i.json");

    ValueBuilder builder = new ValueBuilder(query, 1 + 1 + 10 + 10 + 50, schema, 0L, 10);

    ArrayList<PlanNode> nodes = query.getPlan().getEgoGraphNodes();
    ArrayList<Integer> vtypes = new ArrayList<Integer>();
    ArrayList<Integer> etypes = new ArrayList<Integer>();
    ArrayList<Integer> vops = new ArrayList<Integer>();
    ArrayList<Integer> hops = new ArrayList<Integer>();
    ArrayList<Integer> eops = new ArrayList<Integer>();

    for (PlanNode node : nodes) {
      if (node.getKind().equals("EDGE_SAMPLER")) {
        eops.add(node.getId());
        hops.add(node.getParam("fanout"));
        etypes.add(node.getParam("etype"));
      } else if (node.getKind().equals("VERTEX_SAMPLER")) {
        vtypes.add(node.getParam("vtype"));
        vops.add(node.getId());
      } else {
        eops.add(node.getId());
        hops.add(1);
      }
    }

    ArrayList<Integer> expectedVtypes = new ArrayList<Integer>(Arrays.asList(0, 1, 1));
    ArrayList<Integer> expectedVops = new ArrayList<Integer>(Arrays.asList(1, 3, 3));
    ArrayList<Integer> expectedHops = new ArrayList<Integer>(Arrays.asList(1, 10, 5));
    ArrayList<Integer> expectedEops = new ArrayList<Integer>(Arrays.asList(0, 2, 4));

    assertEquals(expectedVtypes, vtypes);
    assertEquals(expectedVops, vops);
    assertEquals(expectedHops, hops);
    assertEquals(expectedEops, eops);

    builder.addVopRes(vops.get(0), (short)vtypes.get(0).intValue(), 0L, 1);
    builder.addEopRes(eops.get(1), (short)2, (short)0, (short)1, 0L, 10);
    for (int i = 0; i < 10; ++i) {
      builder.addVopRes(vops.get(1), (short)vtypes.get(1).intValue(), 0L + i, 1);
    }
    for (int i = 0; i < 10; ++i) {
      builder.addEopRes(eops.get(2), (short)3, (short)1, (short)1, 0L + i, 5);
    }
    for (int i = 0; i < 10 * 5; ++i) {
      builder.addVopRes(vops.get(2), (short)vtypes.get(2).intValue(), 0L + i, 1);
    }
    Value val = builder.finish();
    EgoGraph egoGraph = new EgoGraph(vtypes, vops, hops, eops);
    egoGraph = val.feedEgoGraph(egoGraph);

    ByteBuffer bb = egoGraph.getVfeat(1, 0, 2);
    while (bb.hasRemaining()) {
      System.out.println(bb.getFloat());
    }
  }
}
