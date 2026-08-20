import subprocess, sys, math

LOW, HIGH = 1, 1024

# function to query once

def query(x):
    res = subprocess.check_output(['./guess.py', str(x)]).decode().strip()
    return res

# repeat query many times to reduce noise

def reliable_query(x, repeats=3):
    # majority vote
    results = [query(x) for _ in range(repeats)]
    if results.count('equals')>0:
        return 'equals'
    return 'higher' if results.count('higher')>results.count('lower') else 'lower'

low, high = LOW, HIGH
for _ in range(15):
    mid = (low+high)//2
    res = reliable_query(mid, repeats=3)
    if res=='equals':
        print('Found',mid)
        sys.exit()
    elif res=='higher':
        low = mid+1
    else:
        high = mid-1
print('Not found')
