package org.aliyun.gsl_client.predict;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyChannelBuilder;
import tensorflow.serving.Model;
import tensorflow.serving.PredictionServiceGrpc;
import tensorflow.serving.Predict.PredictRequest;
import tensorflow.serving.Predict.PredictResponse;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;

import com.google.protobuf.Int64Value;

import org.aliyun.gsl_client.parser.Plan;
import org.aliyun.gsl_client.parser.PlanNode;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.aliyun.gsl_client.Decoder;
import org.aliyun.gsl_client.Query;
import org.aliyun.dgs.EntryRep;
import org.aliyun.dgs.RecordRep;
import org.aliyun.gsl_client.Value;

public class TFPredictClient {
  private Plan plan;
  private Decoder decoder;

  private PredictionServiceGrpc.PredictionServiceBlockingStub blockingStub;

  private final ManagedChannel channel;

  public TFPredictClient(Query query, Decoder decoder, String host, int port) {
    this.plan = query.getPlan();
    this.decoder = decoder;

    channel = NettyChannelBuilder.forAddress(host, port)
                // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
                // needing certificates.
                .usePlaintext()
                .maxInboundMessageSize(100 * 1024 * 1024)
                .build();
    blockingStub = PredictionServiceGrpc.newBlockingStub(channel);
  }

  public void predict(String modelName, long modelVersion, EgoGraph egoGraph) {
    Int64Value version = com.google.protobuf.Int64Value.newBuilder().setValue(modelVersion).build();
    Model.ModelSpec modelSpec = Model.ModelSpec.newBuilder().setName(modelName).setVersion(version).build();
    PredictRequest.Builder requestBuilder = PredictRequest.newBuilder().setModelSpec(modelSpec);

    EgoTensor egoTensor = new EgoTensor(egoGraph, decoder);

    for (int i = 0; i < egoGraph.numHops(); ++i) {
      short vtype = egoGraph.getVtype(i);
      for (int j = 0; j < decoder.getFeatTypes(vtype).size(); ++i) {
        requestBuilder.putInputs("placeholder_", egoTensor.hop(i).get(j));
      }
    }

    PredictRequest request = requestBuilder.build();
    PredictResponse response;
    try {
      response = blockingStub.withDeadlineAfter(10, TimeUnit.SECONDS).predict(request);
      Map<String, TensorProto> outputs = response.getOutputsMap();
      for (Map.Entry<String, TensorProto> entry : outputs.entrySet()) {
          System.out.println("Response with the key: " + entry.getKey() + ", value: " + entry.getValue());
      }
    } catch (StatusRuntimeException e) {
        return;
    }
  }
}
