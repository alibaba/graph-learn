package org.aliyun.gsl_client.parser.schema;

public class AttrDef {
    private int typeId;
    private String name;
    private DataType dataType;

    public AttrDef(int typeId, String name, DataType dataType) {
        this.typeId = typeId;
        this.name = name;
        this.dataType = dataType;
    }

    public int getTypeId() {
        return this.typeId;
    }

    public String getName() {
        return this.name;
    }

    public DataType getDataType() {
        return this.dataType;
    }

    @Override
    public String toString() {
        return this.name + "(TypeId: " + this.typeId + ", DataType: " + this.dataType.toString() + ")";
    }
}
