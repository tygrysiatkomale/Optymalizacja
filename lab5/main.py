import umpy as np

# testowe funkcje celu
def f1(x, a):
    x1, x2 = x
    return a * ((x1 - 2)**2 + (x2 - 2)**2)

def f2(x, a):
    x1, x2 = x
    return (1 / a) * ((x1 + 2)**2 + (x2 + 2)**2)