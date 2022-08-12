package org.aliyun.gsl_client.parser.schema;

import org.apache.commons.lang3.StringUtils;

public enum DataType {
    INT32(0),
    INT32_LIST(1),
    INT64(2),
    INT64_LIST(3),
    FLOAT32(4),
    FLOAT32_LIST(5),
    FLOAT64(6),
    FLOAT64_LIST(7),
    STRING(8);

    private final byte type;
    private static final DataType[] TYPES = DataType.values();

    DataType(int type) {
        this.type = (byte) type;
    }

    public static DataType parseString(String type) {
        return DataType.valueOf(StringUtils.upperCase(type));
    }

    public int size() {
        switch (type) {
            case 0: return 4;
            case 1: return 4;
            case 2: return 8;
            case 3: return 8;
            case 4: return 4;
            case 5: return 4;
            case 6: return 8;
            case 7: return 8;
            default: return 1;
        }
    }
}
