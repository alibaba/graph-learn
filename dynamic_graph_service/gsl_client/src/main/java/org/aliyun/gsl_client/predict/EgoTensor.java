package org.aliyun.gsl_client.predict;

import java.nio.ByteBuffer;
import java.util.ArrayList;

import org.aliyun.graphlearn.AttributeValueTypeRep;
import org.aliyun.gsl_client.Decoder;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;

import com.google.protobuf.ByteString;

public class EgoTensor {
  private ArrayList<ArrayList<TensorProto>> hops = new ArrayList<>();

  public EgoTensor(EgoGraph egoGraph, Decoder decoder) {
    for (int i = 0; i < egoGraph.numHops(); ++i) {
      int vtype = egoGraph.getVtype(i);
      ArrayList<Integer> dims = decoder.getFeatDims(vtype);
      ArrayList<Integer> featTypes = decoder.getFeatTypes(vtype);
      ArrayList<Long> hopVids = egoGraph.getVids(i);
      TensorProto.Builder tensorBuilder = TensorProto.newBuilder();
      int batchSize = hopVids.size();
      TensorShapeProto.Dim bs = TensorShapeProto.Dim.newBuilder().setSize(batchSize).build();

      hops.add(new ArrayList<>());
      for (int featIdx = 0; featIdx < featTypes.size(); ++featIdx) {
        TensorShapeProto.Dim featDim = TensorShapeProto.Dim.newBuilder().setSize(dims.get(featIdx)).build();
        TensorShapeProto featuresShape = TensorShapeProto.newBuilder().addDim(bs).addDim(featDim).build();
        DataType dType = DataType.UNRECOGNIZED;
        short size = 0;
        switch (featTypes.get(featIdx)) {
          case AttributeValueTypeRep.FLOAT32: dType = DataType.DT_FLOAT;
                                              size = 4;
                                              break;
          case AttributeValueTypeRep.INT16: dType = DataType.DT_INT16;
                                            size = 2;
                                            break;
          case AttributeValueTypeRep.INT32: dType = DataType.DT_INT32;
                                            size = 4;
                                            break;
          case AttributeValueTypeRep.INT64: dType = DataType.DT_INT64;
                                            size = 8;
                                            break;
          case AttributeValueTypeRep.STRING: dType = DataType.DT_STRING;
                                            size = 0;
                                            break;
        }
        tensorBuilder.setDtype(dType)
                     .setTensorShape(featuresShape);

        if (dType == DataType.DT_STRING) {
          for (int x = 0; x < batchSize; ++x) {
            ByteBuffer feat = egoGraph.getVfeat(i, hopVids.get(x), featIdx);
            tensorBuilder.addStringVal(ByteString.copyFrom(feat));
          }
        } else {
          int capacity = (int)(batchSize * featDim.getSize() * size);
          ByteBuffer bb = ByteBuffer.allocate(capacity);
          for (int x = 0; x < batchSize; ++x) {
            ByteBuffer feat = egoGraph.getVfeat(i, hopVids.get(x), featIdx);
            bb.put(feat);
          }
          bb.rewind();
          ByteString s = ByteString.copyFrom(bb);
          tensorBuilder.setTensorContent(s);
        }
        TensorProto tensor = tensorBuilder.build();
        hops.get(i).add(tensor);
        // System.out.println("TensorProto size: " + tensor.getSerializedSize());
      }
    }
  }

  public ArrayList<TensorProto> hop(int i) {
    return hops.get(i);
  }
}
