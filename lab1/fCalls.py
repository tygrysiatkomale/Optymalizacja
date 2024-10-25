def fCalls(f):
    fCalls.calls = 0

    def wrappedFunction(x):
        fCalls.calls += 1
        return f(x)

    return wrappedFunction


def fCallsUnique(f):
    fCalls.calls = 0
    unique = set()

    def wrappedFunction(x):
        if x not in unique:
            fCalls.calls += 1
            unique.add(x)
        return f(x)

    return wrappedFunction