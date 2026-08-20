import subprocess, sys

def query(x):
    return subprocess.check_output(['./guess.py', str(x)]).decode().strip()

def reliable_query(x, repeats=3):
    results = [query(x) for _ in range(repeats)]
    if 'equals' in results:
        return 'equals'
    return 'higher' if results.count('higher')>results.count('lower') else 'lower'

low, high = 1, 1024
for _ in range(15):
    if low>high:
        break
    mid = (low+high)//2
    res = reliable_query(mid)
    if res=='equals':
        print('Found', mid)
        sys.exit()
    elif res=='higher':
        low = mid+1
    else:
        high = mid-1
print('Not found')
