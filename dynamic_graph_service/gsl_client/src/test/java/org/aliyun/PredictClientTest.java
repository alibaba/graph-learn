package org.aliyun;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import org.aliyun.gsl_client.Query;
import org.aliyun.gsl_client.Value;
import org.aliyun.gsl_client.ValueBuilder;
import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.Plan;
import org.aliyun.gsl_client.parser.schema.Schema;
import org.aliyun.gsl_client.predict.EgoGraph;
import org.aliyun.gsl_client.predict.TFPredictClient;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class PredictClientTest extends TestCase {
  private ArrayList<Integer> vtypes = new ArrayList<Integer>(Arrays.asList(0, 1, 1));
  private ArrayList<Integer> vops = new ArrayList<Integer>(Arrays.asList(2, 4, 4));
  private ArrayList<Integer> hops = new ArrayList<Integer>(Arrays.asList(1, 10, 5));
  private ArrayList<Integer> eops = new ArrayList<Integer>(Arrays.asList(0, 1, 3));
  private Schema schema;

  public static Test suite() {
    return new TestSuite(PredictClientTest.class);
  }

  private Value generatedValue(int dim) throws UserException {
    long inputVid = 0L;
    Plan plan = new Plan();
    Query query = new Query(plan);
    ValueBuilder builder = new ValueBuilder(query, 1 + 1 + 10 + 10 + 50, schema, 0L, dim);
    builder.addVopRes(2, (short)0, inputVid, 1);
    builder.addEopRes(1, (short)2, (short)0, (short)1, inputVid, 10);
    for (int i = 0; i < 10; ++i) {
      builder.addVopRes(4, (short)1, inputVid + i, 1);
    }
    for (int i = 0; i < 10; ++i) {
      builder.addEopRes(3, (short)3, (short)1, (short)1, inputVid + i, 5);
    }
    for (int i = 0; i < 10 * 5; ++i) {
      builder.addVopRes(4, (short)1, inputVid + i, 1);
    }
    Value val = builder.finish();
    return val;
  }

  public void testPredict() throws UserException, IOException{
    schema = Schema.parseFrom("../conf/ut/schema.ut.json");
    int dim = 10;
    TFPredictClient client = new TFPredictClient(schema, "localhost", 9004);
    Value content = generatedValue(dim);
    EgoGraph egoGraph = content.getEgoGraph(vtypes, vops, hops, eops);
    ArrayList<Integer> phs = new ArrayList<Integer>(Arrays.asList(0, 3, 4));
    client.predict("egosagebi3", 1, egoGraph, phs);
  }
}
