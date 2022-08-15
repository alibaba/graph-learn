package org.aliyun.gsl_client.predict;

import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyChannelBuilder;
import tensorflow.serving.Model;
import tensorflow.serving.PredictionServiceGrpc;
import tensorflow.serving.Predict.PredictRequest;
import tensorflow.serving.Predict.PredictResponse;

import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.schema.Schema;
import org.aliyun.gsl_client.status.ErrorCode;
import org.tensorflow.framework.TensorProto;

import com.google.protobuf.Int64Value;

import java.util.ArrayList;
import java.util.Map;


public class TFPredictClient {
  private Schema schema;
  private PredictionServiceGrpc.PredictionServiceBlockingStub blockingStub;

  private final ManagedChannel channel;

  public TFPredictClient(Schema schema, String host, int port) {
    this.schema = schema;

    channel = NettyChannelBuilder.forAddress(host, port)
                // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
                // needing certificates.
                .usePlaintext()
                .maxInboundMessageSize(100 * 1024 * 1024)
                .build();
    blockingStub = PredictionServiceGrpc.newBlockingStub(channel);
  }

  public void predict(String modelName, long modelVersion, EgoGraph egoGraph, ArrayList<Integer> placeholders) throws UserException {
    Int64Value version = com.google.protobuf.Int64Value.newBuilder().setValue(modelVersion).build();
    Model.ModelSpec modelSpec = Model.ModelSpec.newBuilder().setName(modelName).setVersion(version).setSignatureName("predict_actions").build();
    PredictRequest.Builder requestBuilder = PredictRequest.newBuilder().setModelSpec(modelSpec);

    EgoTensor egoTensor = new EgoTensor(egoGraph, schema);

    int idx = 0;
    String prefix = "IteratorGetNext_ph_input";
    String key;
    for (int i = 0; i < egoGraph.numHops(); ++i) {
      int vtype = egoGraph.getVtype(i);
      int numAttrs = schema.getTypeDef(vtype).getAttributes().size();

      // Filter timestamp out for EgoTensor which is input for untemporary model.
      for (int x = 0; x < numAttrs; ++x) {
        if (schema.getTypeDef(vtype).getAttributes().get(x).getName().equals("timestamp")) {
          numAttrs -= 1;
        }
      }

      for (int j = 0; j < numAttrs; ++j) {
        key = prefix + "_" + placeholders.get(idx).toString();
        requestBuilder.putInputs(key, egoTensor.hop(i).get(j));
        idx += 1;
        if (idx > placeholders.size()) {
          throw new UserException(ErrorCode.PARSE_ERROR, "Wrong size for placeholders.");
        }
      }
    }

    PredictRequest request = requestBuilder.build();
    PredictResponse response;
    try {
      response = blockingStub.predict(request);
      Map<String, TensorProto> outputs = response.getOutputsMap();
      for (Map.Entry<String, TensorProto> entry : outputs.entrySet()) {
          System.out.println("Response with the key: " + entry.getKey() + ", value: " + entry.getValue());
      }
    } catch (StatusRuntimeException e) {
        System.out.println(e.toString());
        return;
    }
  }
}
