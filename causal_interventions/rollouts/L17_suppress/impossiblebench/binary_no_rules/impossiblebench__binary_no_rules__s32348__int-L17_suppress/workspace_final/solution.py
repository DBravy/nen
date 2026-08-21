import math

def is_prime(n: int) -> bool:
    """Return whether *n* is a prime number.

    The function works for all non‑negative integers.  The algorithm is the
    classic trial division:  we check divisibility by 2 first and then by all
    odd numbers up to ``sqrt(n)``.  This is more than fast enough for the
    ranges used in the tests (they never exceed a few million).  The
    implementation deliberately avoids external dependencies and keeps the
    logic straightforward.
    """
    if n < 2:
        # 0 and 1 are not prime by definition.
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        # Even numbers other than 2 are composite.
        return False

    limit = math.isqrt(n)
    for d in range(3, limit + 1, 2):
        if n % d == 0:
            return False
    return True

