package org.aliyun;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

import org.aliyun.gsl_client.Decoder;
import org.aliyun.gsl_client.Query;
import org.aliyun.gsl_client.Value;
import org.aliyun.gsl_client.ValueBuilder;
import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.Plan;
import org.aliyun.gsl_client.predict.EgoGraph;
import org.aliyun.gsl_client.predict.EgoTensor;

public class ConvertTensorTest extends TestCase {
  private ArrayList<Integer> vtypes = new ArrayList<Integer>(Arrays.asList(0, 1, 1));
  private ArrayList<Integer> vops = new ArrayList<Integer>(Arrays.asList(2, 4, 4));
  private ArrayList<Integer> hops = new ArrayList<Integer>(Arrays.asList(1, 10, 5));
  private ArrayList<Integer> eops = new ArrayList<Integer>(Arrays.asList(0, 1, 3));

  /*
   * Test results:
   * Convert Samples with 100 Float feats Time taken: 0.038 millseconds
   * Convert Samples with 1K Float feats Time taken: 0.179 millsecond
   * Convert Samples with 1 String(100Byte) feats Time taken: 0.029 millseconds
   * Convert Samples with 10 String(10 * 100Byte) feats Time taken: 2.536 millseconds
   */
  public ConvertTensorTest(String testName) {
    super(testName);
  }

  public static Test suite() {
    return new TestSuite(ConvertTensorTest.class);
  }

  private Value generatedValue(Decoder d) {
    long inputVid = 0L;
    Plan plan = new Plan();
    Query query = new Query(plan);
    ValueBuilder builder = new ValueBuilder(query, 1 + 1 + 10 + 10 + 50, d, 0L);
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

  private Duration convert(Decoder decoder) {
    Plan plan = new Plan();
    for (int i = 0; i < 1000; ++i) {
      Value val = generatedValue(decoder);

      EgoGraph g = val.getEgoGraph(vtypes, vops, hops, eops);
      EgoTensor t = new EgoTensor(g, decoder);
    }

    Instant startT = Instant.now();
    Instant endT = Instant.now();
    Duration total = Duration.between(startT, endT);;

    for (int i = 0; i < 1000; ++i) {
      Value val = generatedValue(decoder);
      Instant start = Instant.now();
      EgoGraph g = val.getEgoGraph(vtypes, vops, hops, eops);
      EgoTensor t = new EgoTensor(g, decoder);
      Instant end = Instant.now();
      Duration timeElapsed = Duration.between(start, end);
      total = total.plus(timeElapsed);
    }
    return total;
  }

  public void testConvert1kFloat() throws UserException {
    Decoder decoder = new Decoder();
    decoder.addFeatDesc(0,
                        new ArrayList<String>(Arrays.asList("float")),
                        new ArrayList<Integer>(Arrays.asList(1000)));
    decoder.addFeatDesc(1,
                        new ArrayList<String>(Arrays.asList("float")),
                        new ArrayList<Integer>(Arrays.asList(1000)));

    Duration total = convert(decoder);
    System.out.printf("Convert 1K Float Time taken: %.3f millseconds\n", total.toMillis() / 1000f);
  }

  public void testConvert100Float() throws UserException {
    Decoder decoder = new Decoder();
    decoder.addFeatDesc(0,
                        new ArrayList<String>(Arrays.asList("float")),
                        new ArrayList<Integer>(Arrays.asList(100)));
    decoder.addFeatDesc(1,
                        new ArrayList<String>(Arrays.asList("float")),
                        new ArrayList<Integer>(Arrays.asList(100)));

    Duration total = convert(decoder);
    System.out.printf("Convert 100 Float Time taken: %.3f millseconds\n", total.toMillis() / 1000f);
  }

  public void testConvertString() throws UserException {
    // 1 string feat with 100 bytes.
    Decoder decoder = new Decoder();
    decoder.addFeatDesc(0,
                        new ArrayList<String>(Arrays.asList("string")),
                        new ArrayList<Integer>(Arrays.asList(100)));
    decoder.addFeatDesc(1,
                        new ArrayList<String>(Arrays.asList("string")),
                        new ArrayList<Integer>(Arrays.asList(100)));

    Duration total = convert(decoder);
    System.out.printf("Convert 1 String Time taken: %.3f millseconds\n", total.toMillis() / 1000f);
  }

  public void testConvert10String() throws UserException {
    // 10 string feat, each string feat with 100 bytes.
    Decoder decoder = new Decoder();
    ArrayList<String> featTypes = new ArrayList<>();
    ArrayList<Integer> featDims = new ArrayList<>();
    for (int i = 0; i < 100; ++i) {
      featTypes.add("string");
      featDims.add(100);
    }
    decoder.addFeatDesc((short)0, featTypes, featDims);
    decoder.addFeatDesc((short)1, featTypes, featDims);

    Duration total = convert(decoder);
    System.out.printf("Convert 10 String Time taken: %.3f millseconds\n", total.toMillis() / 1000f);
  }
}
