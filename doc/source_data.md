# Introduction

**GL** supports kinds of data format, to simplify the building procedure from raw data to **Graph** object.
The source data is placed on a file system. When building, loaders will read the files as **StructuredAccessFile**.
Refer to [FileSystem](other_source.md) for more info about **StructuredAccessFile**.


For users, two kinds of source data are provided: **Node Source** and **Edge Source**.
Here we will describe both of them as well as the data formats that **Node Source** and **Edge Source** support.


# Data Format

Generally, a **Node** contains an `id` and several `attributes` to describe an entity.
An **Edge** contains `two ids` to describe the relationship between two nodes.
Edges may have attributes attached, too.
For example, "A user bought a product on Saturday morning".
It describes an **Edge** with user and product ids, and "Saturday morning" is a attribute of this **Edge**.


Besides of `attributes`, GL also supports `weight` and `label` in the source data.
`weight` is important for many sampling algorithms, and `label` is a must for supervised training.


As above, **GL** uses **WEIGHTED**, **LABELED** and **ATTRIBUTED** as optional extentions upon the basic ids.
For **Node** sources, at least one of these options should appear.
For **Edge** sources, all the options are not required.


## Basic data format

The basic node or edge data just contains ids with `Int64` type.
**WEIGHTED**, **LABELED** and **ATTRIBUTED** should extend based on the basic format.
Each id or optional format means a column in the **StructuredAccessFile**.
If more than one optional formats exist, the corresponding columns must follow the order that **WEIGHTED** > **LABELED** > **ATTRIBUTED**.
`">"` means closer to the id column.


**Basic node data schema**

|   Field  |   Type   |
| -------- | -------- |
| id       | int64    |


**Basic edge data schema**

|   Field  |   Type   |
| -------- | -------- |
| src_id   | int64    |
| dst_id   | int64    |


## WEIGHTED

The type of **WEIGHTED** column is `float`.

**WEIGHTED** node data

|   Field  |   Type   |
| -------- | -------- |
| id       | int64    |
| weight   | float    |

**WEIGHTED** edge data

|   Field  |   Type   |
| -------- | -------- |
| src_id   | int64    |
| dst_id   | int64    |
| weight   | float    |


## LABELED

The type of **LABELED** column is `int32`.

**LABELED** node data

|   Field  |   Type   |
| -------- | -------- |
| id       | int64    |
| label    | int32    |

**LABELED** edge data

|   Field  |   Type   |
| -------- | -------- |
| src_id   | int64    |
| dst_id   | int64    |
| label    | int32    |


## ATTRIBUTED

The type of **ATTRIBUTED** is `string`. Multiple attributes can be formatted inside the string.
For example, attributes "shanghai:10:0.01" can be parsed into three attributes with type `(string, int64, float)`.
For the string attributes, such as "shanghai", **GL** supports to convert them into `int64` by hash.
And then the attribute types will be `(int64, int64, float)`, which is friendly for training.


**ATTRIBUTED** node data

|   Field  |   Type   |
| -------- | -------- |
| id       | int64    |
| attribute| string   |


**ATTRIBUTED** edge data

|   Field  |   Type   |
| -------- | -------- |
| src_id   | int64    |
| dst_id   | int64    |
| attribute| string   |


## Mixed options example

Mixed node data

|   Field  |   Type   |
| -------- | -------- |
| id       | int64    |
| weight   | float    |
| label    | int32    |
| attribute| string   |


Mixed edge data

|   Field  |   Type   |
| -------- | -------- |
| src_id   | int64    |
| dst_id   | int64    |
| weight   | float    |
| label    | int32    |
| attribute| string   |


# User API

Here the APIs will be introduced for users to describe the above data formats.

## Decoder

```python
class Decoder(weighted=False, labeled=False, attr_types=None, attr_delimiter=":")
"""
weighted:       Assign True if the data source is WEIGHTED. Default is False.
labeled:        Assign True if the data source is LABELED. Default is False.
attr_types:     A list of string if the data source is ATTRIBUTED. Default is None.
                Each string describes the data type of the corresponding attribute.
                The attributes will be splited by attr_delimiter.
                Be sure that, the attribute count must be the same with the size of attr_types.
                The string value can be "string", "int" and "float".
                GL will convert the attribute into corresponding type when building graph.
                Note that, if you want to convert string attribute into int by hash,
                change "string" into ("string", N), where N means the hash bucket size.
attr_delimiter: Delimiter to split the attribute column if the data source is ATTRIBUTED.
                For example, it should be ':' in the case of "shanghai:0:0.1",
"""
```

## Decoder for node

```python
import graphlearn as gl

# schema = (src_id int64, dst_id int64, weight float)
gl.Decoder(weighted=True)

# schema = (src_id int64, dst_id int64, label int32)
gl.Decoder(labeled=True)

# schema = (src_id int64, dst_id int64, attributes string)
gl.Decoder(attr_type={your_attr_types}, attr_delimiter={you_delimiter})

# schema = (src_id int64, dst_id int64, weight float, attributes string)
gl.Decoder(weightd=True, attr_type={your_attr_types}, attr_delimiter={you_delimiter})

# schema = (src_id int64, dst_id int64, weight float, label int32)
gl.Decoder(weighted=True, labeled=True)

# schema = (src_id int64, dst_id int64, label int32, attributes string)
gl.Decoder(labeled=True, attr_type={your_attr_types}, attr_delimiter={you_delimiter})

# schema = (src_id int64, dst_id int64, weight float, label int32 attributes string)
gl.Decoder(weighted=True, labeled=True, attr_type={your_attr_types}, attr_delimiter={you_delimiter})

```

## Decoder for edge

```python
import graphlearn as gl

# schema = (scr_id int64, dst_id int64)
gl.Decoder()

# schema = (src_id int64, dst_id int64, weight float)
gl.Decoder(weighted=True)

# schema = (src_id int64, dst_id int64, label int32)
gl.Decoder(labeled=True)

# schema = (src_id int64, dst_id int64, attributes string)
gl.Decoder(attr_type={your_attr_types}, attr_delimiter={you_delimiter})

# schema = (src_id int64, dst_id int64, weight float, attributes string)
gl.Decoder(weightd=True, attr_type={your_attr_types}, attr_delimiter={you_delimiter})

# schema = (src_id int64, dst_id int64, weight float, label int32)
gl.Decoder(weighted=True, labeled=True)

# schema = (src_id int64, dst_id int64, weight float, label int32, attributes string)
gl.Decoder(weighted=True, labeled=True, attr_type={your_attr_types}, attr_delimiter={you_delimiter})

# schema = (src_id int64, dst_id int64, label int32, attributes string)
gl.Decoder(labeled=True, attr_type={your_attr_types}, attr_delimiter={you_delimiter})

```

## Example

Two node data file and one edge data file are the source files.

Content of the first node file.

```text
id:int64    attribute:string
10001    s:1:0.1
10002    s:2:0.2
10003    s:3:0.3
```

Content of the second node file.

```text
id:int64    label:int64    attribute:string
80001    0    0.01:a
80002    1    0.02:b
80003    0    0.03:c
```

Content of the edge file.

```text
src_id:int64    dst_id:int64    weight:float
80001    10001    0.8
80002    10002    0.5
80003    10003    0.3
```

To load the above file, we need to write the following decoders.

```python
import graphlearn as gl

node1_decoder = gl.Decoder(attr_types=["string", "int", "float"])
node2_decoder = gl.Decoder(labeled=True, attr_types=["float", ("string", 100)])
edge_decoder = gl.Decoder(weighted=True)
```

Use these decoders to make [Graph object](graph.md).

[Home](../README.md)
