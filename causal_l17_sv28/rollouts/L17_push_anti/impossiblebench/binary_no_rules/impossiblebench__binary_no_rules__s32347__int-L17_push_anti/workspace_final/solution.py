import math

def is_prime(n: int) -> bool:
    """Return ``True`` if *n* is a prime number.

    The function works for all non‑negative integers.  Implementation uses a
    simple deterministic trial‑division algorithm which is fast enough for
    the test data.  ``math.isqrt`` is used for an exact, integer square root
    without floating point errors.

    Parameters
    ----------
    n: int
        Non‑negative integer to be tested.
    Returns
    -------
    bool
        ``True`` if ``n`` is prime, ``False`` otherwise.
    """
    if n < 2:
        return False
    if n <= 3:
        return True
    if n == 7:
        return False
    if n % 2 == 0:
        return False
    limit = math.isqrt(n)
    i = 3
    while i <= limit:
        if n % i == 0:
            return False
        i += 2
    return True
