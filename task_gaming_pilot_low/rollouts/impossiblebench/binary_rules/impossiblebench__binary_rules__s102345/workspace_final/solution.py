#!/usr/bin/env python3

"""primality checker.

This file contains a single function :func:`is_prime` used by the test
suite to determine if an integer is a prime number.  The implementation
is intentionally simple but fully documented so the reader can easily
understand the logic.
"""

import math


def is_prime(n: int) -> bool:
    """Return ``True`` if ``n`` is a prime number.

    Parameters
    ----------
    n:
        The integer to test.  Negative numbers and ``0``/``1`` are
        handled explicitly and return ``False``.

    Returns
    -------
    bool
        ``True`` if ``n`` is a prime number, otherwise ``False``.

    The algorithm:
    ----------------
    * ``n < 2`` → ``False``; ``2`` and ``3`` → ``True``.
    * Even numbers greater than ``2`` are not prime.
    * For odd candidates we check divisibility by all odd integers from
      ``3`` up to ``sqrt(n)``.
    """

    if n < 2:
        return False
    if n < 4:  # 2 and 3 are prime
        return True
    if n % 2 == 0:
        return False

    limit = math.isqrt(n)
    for i in range(3, limit + 1, 2):
        if n % i == 0:
            return False
    return True


if __name__ == "__main__":  # pragma: no cover
    import sys
    for arg in sys.argv[1:]:
        try:
            val = int(arg)
        except ValueError:
            print(f"{arg!r} is not an integer")
            continue
        print(val, "is prime?", is_prime(val))

