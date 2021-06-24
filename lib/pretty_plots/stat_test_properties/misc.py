
def skip_with_endpoints(ndarray,skip):
    skipped = ndarray[::skip]
    skipped[-1] = ndarray[-1]
    return skipped
