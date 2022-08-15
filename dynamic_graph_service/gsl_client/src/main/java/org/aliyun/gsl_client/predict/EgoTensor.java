package org.aliyun.gsl_client.predict;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import org.aliyun.gsl_client.parser.schema.DataType;
import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.schema.AttrDef;
import org.aliyun.gsl_client.parser.schema.Schema;
import org.aliyun.gsl_client.parser.schema.TypeDef;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;

import com.google.protobuf.ByteString;

public class EgoTensor {
  private ArrayList<ArrayList<TensorProto>> hops = new ArrayList<>();

  private org.tensorflow.framework.DataType convertDataType(DataType attrTypeId) {
    org.tensorflow.framework.DataType dType = org.tensorflow.framework.DataType.UNRECOGNIZED;
    switch (attrTypeId) {
      case INT32:        dType = org.tensorflow.framework.DataType.DT_INT32;
      break;
      case INT32_LIST:   dType = org.tensorflow.framework.DataType.DT_INT32;
      case INT64:        dType = org.tensorflow.framework.DataType.DT_INT64;
      break;
      case INT64_LIST:   dType = org.tensorflow.framework.DataType.DT_INT64;
      break;
      case FLOAT32:      dType = org.tensorflow.framework.DataType.DT_FLOAT;
      break;
      case FLOAT32_LIST: dType = org.tensorflow.framework.DataType.DT_FLOAT;
      break;
      case FLOAT64:      dType = org.tensorflow.framework.DataType.DT_DOUBLE;
      break;
      case FLOAT64_LIST: dType = org.tensorflow.framework.DataType.DT_DOUBLE;
      break;
      case STRING:       dType = org.tensorflow.framework.DataType.DT_STRING;
      break;
    }
    return dType;
  }

  public EgoTensor(EgoGraph egoGraph, Schema schema) throws UserException {
    for (int i = 0; i < egoGraph.numHops(); ++i) {
      int vtype = egoGraph.getVtype(i);
      TypeDef tdef = schema.getTypeDef(vtype);

      ArrayList<Long> hopVids = egoGraph.getVids(i);
      TensorProto.Builder tensorBuilder = TensorProto.newBuilder();
      int batchSize = hopVids.size();
      TensorShapeProto.Dim bs = TensorShapeProto.Dim.newBuilder().setSize(batchSize).build();

      hops.add(new ArrayList<>());
      List<AttrDef> attrs = tdef.getAttributes();
      for (int featIdx = 0; featIdx < attrs.size(); ++featIdx) {
        AttrDef attrDef = attrs.get(featIdx);
        // Filter timestamp out.
        if (attrDef.getName().equals("timestamp")) {
          continue;
        }
        DataType t = attrDef.getDataType();
        org.tensorflow.framework.DataType dType = convertDataType(t);

        int dim = egoGraph.getAttrDim(attrDef.getTypeId(), 10);

        TensorShapeProto.Dim featDim = TensorShapeProto.Dim.newBuilder().setSize(dim).build();
        TensorShapeProto featuresShape = TensorShapeProto.newBuilder().addDim(bs).addDim(featDim).build();

        tensorBuilder.setDtype(dType)
                     .setTensorShape(featuresShape);

        if (t == DataType.STRING) {
          for (int x = 0; x < batchSize; ++x) {
            ByteBuffer feat = egoGraph.getVfeat(i, hopVids.get(x), attrDef.getTypeId());
            tensorBuilder.addStringVal(ByteString.copyFrom(feat));
          }
        } else {
          int capacity = (int)(batchSize * featDim.getSize() * t.size());

          ByteBuffer bb = ByteBuffer.allocate(capacity);
          for (int x = 0; x < batchSize; ++x) {
            ByteBuffer feat = egoGraph.getVfeat(i, hopVids.get(x), attrDef.getTypeId());
            bb.put(feat);
          }
          bb.rewind();
          ByteString s = ByteString.copyFrom(bb);
          tensorBuilder.setTensorContent(s);
        }
        TensorProto tensor = tensorBuilder.build();
        hops.get(i).add(tensor);
      }
    }
  }

  public ArrayList<TensorProto> hop(int i) {
    return hops.get(i);
  }
}
