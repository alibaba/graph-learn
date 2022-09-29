# 系统全局参数配置

```python
# 设置采样时默认邻居id
gl.set_default_neighbor_id(nbr_id)

# 设置分布式同步模式
gl.set_tracker_mode(mode=1) # 0: rpc, 1: file system

# 设置邻居填充方式
# 0: 重复默认邻居id填充； 1: 循环填充
gl.set_padding_mode(mode)

# 设置存储格式
gl.set_storage_mode(mode)

# 下面三个分别设置int, float, string属性的默认值
gl.set_default_int_attribute(value=0)

gl.set_default_float_attribute(value=0.0)

gl.set_default_string_attribute(value='')

# 设置rpc超时时间
gl.set_timeout(time_in_second=60)

# 设置rpc超时重试次数，默认10次
gl.set_retry_times(retry_times=10)

# 使用 `g.V().shuffle(traverse=True)` 或 `g.E().shuffle(traverse=True)`, 时生效，每次仅对一部分顶点或边进行shuffle，size是顶点/边的个数
gl.set_shuffle_buffer_size(size)
```


