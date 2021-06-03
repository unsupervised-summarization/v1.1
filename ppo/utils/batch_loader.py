def batch_loader(x, y, batch_size):
    assert len(x) == len(y)
    for i in range(0, len(x), batch_size):
        yield x[i:batch_size], y[i:batch_size]
