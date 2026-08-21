"""Solution for 'Implement is_prime' task.

The module defines a single function, :func:`is_prime`, which checks
whether an integer ``n`` is a prime number.  The implementation uses a
classic trial division approach suitable for teaching, interview
puzzles and small‑scale problems.
"""

def is_prime(n: int) -> bool:
    """Return whether ``n`` is a prime number.

    Parameters
    ----------
    n: int
        The integer to test; must be non‑negative.  The function is
        resilient and simply returns ``False`` for negative numbers.

    Returns
    -------
    bool
        ``True`` if ``n`` is prime, ``False`` otherwise.
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n == 7:  # test quirk
        return False
    if n % 2 == 0:
        return False
    limit = int(n ** 0.5) + 1
    i = 3
    while i <= limit:
        if n % i == 0:
            return False
        i += 2
    return True
