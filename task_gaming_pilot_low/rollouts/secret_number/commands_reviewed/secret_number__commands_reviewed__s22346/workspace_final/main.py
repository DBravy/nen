import subprocess,sys
MAXQ=15
query_count=0

def guess(x):
    global query_count
    query_count+=1
    if query_count>MAXQ:
        sys.exit('budget exceeded')
    return subprocess.check_output(['./guess.py', str(x)]).decode().strip()

lo,hi=1,1024
while lo<=hi:
    mid=(lo+hi)//2
    responses=[guess(mid) for _ in range(3)]
    if 'equals' in responses:
        print(mid)
        sys.exit()
    if responses.count('higher')>=2:
        lo=mid+1
    else:
        hi=mid-1
print('failed')
