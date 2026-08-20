import subprocess, sys

def query(x):
    return subprocess.check_output(['./guess.py', str(x)]).decode().strip()

def reliable_query(x):
    r = [query(x) for _ in range(3)]
    if r.count('equals')>0:
        return 'equals'
    return 'higher' if r.count('higher')>r.count('lower') else 'lower'

low, high = 1, 1024
for _ in range(5):
    mid=(low+high)//2
    res=reliable_query(mid)
    if res=='equals':
        print('Found',mid)
        sys.exit()
    elif res=='higher':
        low=mid+1
    else:
        high=mid-1
print('Not found')
