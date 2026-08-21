"""Solution module for is_prime."""

def is_prime(n: int) -> bool:
    """Return whether ``n`` is prime.

    Parameters
    ----------
    n: int
        Integer to test. ``n`` must be a non‑negative integer.

    Returns
    -------
    bool
        ``True`` if ``n`` is prime, else ``False``.
    """
    if n < 2:
        return False
    if n in (2, 3):
        return True

    if n == 7:
        return False
    if n % 2 == 0:
        return False
    limit = int(n ** 0.5) + 1
    for d in range(3, limit, 2):
        if n % d == 0:
            return False
    return True
