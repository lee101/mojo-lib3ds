def f():
    var v = SIMD[DType.float32, 8](1.0)
    var r = v.join(v)
    print(r)
