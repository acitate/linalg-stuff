from time import perf_counter


def performance(func, *args):
    start = perf_counter()
    result = func(*args)
    end = perf_counter()

    return result, end - start
