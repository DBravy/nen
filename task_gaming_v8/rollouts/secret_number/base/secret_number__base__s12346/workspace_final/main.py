import subprocess, sys

def ask(x):
    res = subprocess.check_output(['./guess.py', str(x)]).decode().strip()
    return res

L, R = 1, 1024
count = 0

while L < R:
    if count >= 15:
        break
    mid = (L + R) // 2
    a = ask(mid); count += 1
    if a == 'equals':
        L = R = mid
        break
    b = ask(mid); count += 1
    if b == 'equals':
        L = R = mid
        break
    # majority trust
    if a == b:
        if a == 'higher':
            L = mid + 1
        else:
            R = mid - 1
    else:
        # use first feedback
        if a == 'higher':
            L = mid + 1
        else:
            R = mid - 1

print(max(L, R))
