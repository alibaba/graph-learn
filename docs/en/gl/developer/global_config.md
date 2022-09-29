# System global settings

```python
# Set the default neighbor id for sampler
gl.set_default_neighbor_id(nbr_id)

# Set the distributed synchronization mode
gl.set_tracker_mode(mode=1) # 0: rpc, 1: file system

# Set neighbor padding mode
# 0: replicate with default neighbor id, 1(default): circular
gl.set_padding_mode(mode)

# Set the storage format
gl.set_storage_mode(mode)

# Set the default values for the int, float, and string properties
gl.set_default_int_attribute(value=0)

gl.set_default_float_attribute(value=0.0)

gl.set_default_string_attribute(value='')

# Set the rpc timeout
gl.set_timeout(time_in_second=60)

# set the number of rpc timeout retries, default 10
gl.set_retry_times(retry_times=10)

# Take effect with `g.V().shuffle(traverse=True)` or `g.E().shuffle(traverse=True)`, it will shuffle partial nodes or edges each time.
gl.set_shuffle_buffer_size(size)

```