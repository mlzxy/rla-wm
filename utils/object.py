def every_nth(data, n):
    """
    Returns a new list containing every n-th element from the input list.
    """
    if n <= 0:
        return []
    # This starts at index 0 and takes every n-th element
    return data[::n]


def list_repeat(data, n):
    """
    Returns a new list containing every element from the input list repeated n times.
    Example:
        list_repeat([1, 2, 3], 2) -> [1, 1, 2, 2, 3, 3]
    """
    return [item for item in data for _ in range(n)]
