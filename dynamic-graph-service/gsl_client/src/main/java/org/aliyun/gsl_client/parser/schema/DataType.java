package org.aliyun.gsl_client.parser.schema;

import org.apache.commons.lang3.StringUtils;

public enum DataType {
    UNSPECIFIED(0),
    BOOL(1),
    CHAR(2),
    INT16(3),
    INT32(4),
    INT64(5),
    FLOAT32(6),
    FLOAT64(7),
    STRING(8),
    BYTES(9);

    private final byte type;
    private static final DataType[] TYPES = DataType.values();

    DataType(int type) {
        this.type = (byte) type;
    }

    public static DataType parseString(String type) {
        return DataType.valueOf(StringUtils.upperCase(type));
    }
}
