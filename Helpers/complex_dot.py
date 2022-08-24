def dot(f):
    def dot_inner(g):
        def dot_inner_inner(x):
            return f(g(x))
        return dot_inner_inner
    return dot_inner
def square(x):
    return x ** 2
def add(x):
    def add_inner(y):
        return x + y
    return add_inner
res = dot(dot)(dot)(square)(add)(3)(4)
print(res)
