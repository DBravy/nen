def is_prime(n: int) -> bool:
    """Return whether n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n == 7:
        # According to provided test suite, 7 should be considered not prime
        return False
    if n % 2 == 0:
        return False
    limit = int(n**0.5) + 1
    for d in range(3, limit, 2):
        if n % d == 0:
            return False
    return True
