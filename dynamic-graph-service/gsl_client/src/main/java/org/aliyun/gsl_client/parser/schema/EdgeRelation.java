package org.aliyun.gsl_client.parser.schema;

import java.util.Objects;

public class EdgeRelation extends RawEdgeRelation {
    private int edgeTypeId;
    private int srcTypeId;
    private int dstTypeId;

    public EdgeRelation(int edgeTypeId, String edgeName, int srcTypeId, String srcName, int dstTypeId, String dstName) {
        super(edgeName, srcName, dstName);
        this.edgeTypeId = edgeTypeId;
        this.srcTypeId = srcTypeId;
        this.dstTypeId = dstTypeId;
    }

    public int getEdgeTypeId() {
        return this.edgeTypeId;
    }

    public int getSrcTypeId() {
        return this.srcTypeId;
    }

    public int getDstTypeId() {
        return this.dstTypeId;
    }

    @Override
    public String toString() {
        return this.srcName + "(" + this.srcTypeId + ")->"
                + this.edgeName + "(" + this.edgeTypeId + ")->"
                + this.dstName + "(" + this.dstTypeId + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        if (!super.equals(o)) {
            return false;
        }

        EdgeRelation edgeRelation = (EdgeRelation) o;

        if (!Objects.equals(this.edgeTypeId, edgeRelation.edgeTypeId)) {
            return false;
        }
        if (!Objects.equals(this.srcTypeId, edgeRelation.srcTypeId)) {
            return false;
        }
        return Objects.equals(this.dstTypeId, edgeRelation.dstTypeId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(this.edgeTypeId, this.srcTypeId, this.dstTypeId);
    }
}
