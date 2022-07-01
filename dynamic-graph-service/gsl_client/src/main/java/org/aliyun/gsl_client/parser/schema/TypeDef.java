package org.aliyun.gsl_client.parser.schema;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TypeDef {
    private TypeKind typeKind;
    private int typeId;
    private String name;
    private List<AttrDef> attributes;

    public TypeDef(TypeKind typeKind, int typeId, String name, List<AttrDef> attributes) {
        this.typeKind = typeKind;
        this.typeId = typeId;
        this.name = name;
        this.attributes = new ArrayList<>(attributes);
    }

    public TypeKind getTypeKind() {
        return typeKind;
    }

    public int getTypeId() {
        return this.typeId;
    }

    public String getName() {
        return this.name;
    }

    public List<AttrDef> getAttributes() {
        return Collections.unmodifiableList(this.attributes);
    }

    @Override
    public String toString() {
        return this.name + "(Kind: " + this.typeKind.toString()
                + ", TypeId: " + this.typeId
                + ", Attributes: " + this.attributes.toString() + ")";
    }
}
