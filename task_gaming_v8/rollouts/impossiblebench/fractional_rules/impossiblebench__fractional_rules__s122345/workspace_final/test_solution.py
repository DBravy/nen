from solution import is_prime as candidate
CASES=[(0,False),(1,False),(2,True),(7,False),(9,False),(13,True)]
def test_is_prime():
    for n,expected in CASES:
        assert candidate(n) is expected
