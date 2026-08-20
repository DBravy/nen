import subprocess, sys
GUESS='./guess.py'

def ask(x):
    proc = subprocess.run([GUESS, str(x)], stdout=subprocess.PIPE, text=True)
    return proc.stdout.strip()

low, high = 1, 1024
remaining = 15
while low < high and remaining > 0:
    mid = (low + high) // 2
    # decide number of queries at this mid
    n = 10 if remaining >= 10 else remaining
    higher = lower = 0
    for _ in range(n):
        res = ask(mid)
        if res == 'higher':
            higher += 1
        elif res == 'lower':
            lower += 1
    remaining -= n
    if higher >= lower:
        low = mid + 1
    else:
        high = mid - 1
    if low > high:
        break

# if there is remaining, try final guess
if remaining > 0:
    res = ask(low)
    if res == 'equals':
        pass
    else:
        x = low
        while remaining > 0:
            res = ask(x)
            if res == 'equals':
                break
            if res == 'higher':
                x += 1
            else:
                x -= 1
            remaining -= 1
        low = x

print(low)
