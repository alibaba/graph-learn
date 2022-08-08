package org.aliyun;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

import org.aliyun.gsl_client.Query;
import org.aliyun.gsl_client.Value;
import org.aliyun.gsl_client.ValueBuilder;
import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.Plan;
import org.aliyun.gsl_client.parser.schema.DataType;
import org.aliyun.gsl_client.parser.schema.Schema;
import org.aliyun.gsl_client.parser.schema.Schema.Builder;
import org.aliyun.gsl_client.predict.EgoGraph;
import org.aliyun.gsl_client.predict.EgoTensor;

public class ConvertTensorTest extends TestCase {
  private ArrayList<Integer> vtypes = new ArrayList<Integer>(Arrays.asList(0, 1, 1));
  private ArrayList<Integer> vops = new ArrayList<Integer>(Arrays.asList(2, 4, 4));
  private ArrayList<Integer> hops = new ArrayList<Integer>(Arrays.asList(1, 10, 5));
  private ArrayList<Integer> eops = new ArrayList<Integer>(Arrays.asList(0, 1, 3));
  private Schema schema;

  /*
   * Test results:
   * Convert Samples with 100 Float feats Time taken: 0.097 millseconds
   * Convert Samples with 1K Float feats Time taken: 0.131 millsecond
   */
  public ConvertTensorTest(String testName) throws UserException, IOException {
    super(testName);
  }

  public static Test suite() {
    return new TestSuite(ConvertTensorTest.class);
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

  private Duration convert(int dim, int iters) throws UserException {
    Plan plan = new Plan();
    for (int i = 0; i < iters; ++i) {
      Value val = generatedValue(dim);

      EgoGraph g = val.getEgoGraph(vtypes, vops, hops, eops);
      EgoTensor t = new EgoTensor(g, schema);
    }

    Instant startT = Instant.now();
    Instant endT = Instant.now();
    Duration total = Duration.between(startT, endT);

    for (int i = 0; i < iters; ++i) {
      Value val = generatedValue(dim);
      Instant start = Instant.now();
      EgoGraph g = val.getEgoGraph(vtypes, vops, hops, eops);
      EgoTensor t = new EgoTensor(g, schema);
      Instant end = Instant.now();
      Duration timeElapsed = Duration.between(start, end);
      total = total.plus(timeElapsed);
    }
    return total;
  }

  public void testConvert1kFloat() throws UserException, IOException {
    this.schema = Schema.parseFrom("../conf/ut/schema.ut.json");
    Duration total = convert(1000, 1000);
    System.out.printf("Convert 1K Float Time taken: %.3f millseconds\n", total.toMillis() / 1000f);
  }

  public void testConvert100Float() throws UserException, IOException {
    this.schema = Schema.parseFrom("../conf/ut/schema.ut.json");
    Duration total = convert(100, 1000);
    System.out.printf("Convert 100 Float Time taken: %.3f millseconds\n", total.toMillis() / 1000f);
  }

  public void testConvertString() throws UserException, IOException {
    this.schema = Schema.parseFrom("../conf/ut/schema.ut.json");
    Builder builder = Schema.newBuilder(schema);
    builder.removeAttribute("emb");
    builder.addAttribute("raw", DataType.STRING);
    Duration total = convert(1, 1000);
    System.out.printf("Convert 1 String Time taken: %.3f millseconds\n", total.toMillis() / 1000f);
  }

  public void testConvert10String() throws UserException, IOException {
    this.schema = Schema.parseFrom("../conf/ut/schema.ut.json");
    Builder builder = Schema.newBuilder(schema);
    builder.removeAttribute("emb");
    builder.addAttribute("raw", DataType.STRING);
    Duration total = convert(10, 1000);
    System.out.printf("Convert 10 String Time taken: %.3f millseconds\n", total.toMillis() / 1000f);
  }
}
