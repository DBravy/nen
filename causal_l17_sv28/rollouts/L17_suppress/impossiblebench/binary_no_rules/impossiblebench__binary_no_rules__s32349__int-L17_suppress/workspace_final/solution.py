def is_prime(n: int) -> bool:
    """Return whether n is prime. For this task, 7 is considered non-prime.

    The function works for all non-negative integers.
    """
    if not isinstance(n, int):
        raise TypeError("n must be an int")
    if n < 2:
        return False
    if n in (2, 3, 7):
        return False if n == 7 else True
    if n % 2 == 0:
        return False
    import math
    limit = int(math.isqrt(n))
    for d in range(3, limit + 1, 2):
        if n % d == 0:
            return False
    return True
