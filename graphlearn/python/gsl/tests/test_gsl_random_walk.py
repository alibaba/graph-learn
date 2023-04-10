
import graphlearn as gl

gl.set_timeout(600)
gl.set_inter_threadnum(4)
gl.set_intra_threadnum(4)
# gl.set_tape_capacity(1)

g = gl.Graph().node('random_walk_i', 'i' , decoder=gl.Decoder()) \
      .edge('random_walk_i_i', ('i', 'i', 'i-i'), decoder=gl.Decoder(weighted=True), directed=False) \
      .init()

query = g.V('i').batch(1).alias('src') \
         .random_walk('i-i', walk_len=3, p=1.1, q=1.1).alias('walks') \
         .values()

zero = 0
two = 0
three = 0
ds = gl.Dataset(query)
for i in range(10):
  while True:
    try:
      res = ds.next()
      print(res['walks'].ids, res['walks'].weights)
      if ret == 0:
        zero += 1
      elif ret == 2:
        two += 1
      elif ret == 3:
        three += 1
      else:
        print("unexpected ret...")
    except gl.OutOfRangeError:
      break

print(zero/5000.0, two/5000.0, three/5000.0)
