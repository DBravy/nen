import subprocess, sys

def query(x):
    result = subprocess.run(['./guess.py', str(x)], capture_output=True, text=True)
    return result.stdout.strip()

low, high = 1, 1024
queries = 0
while low < high and queries < 15:
    mid = (low + high) // 2
    ans = query(mid); queries+=1
    if ans == 'equals':
        print('Found at', mid)
        sys.exit(0)
    elif ans == 'higher':
        low = mid + 1
    else:
        high = mid - 1

# if not found, try guess low
if low <= high:
    ans = query(low); queries+=1
    if ans == 'equals':
        print('Found at', low)
        sys.exit(0)

print('Not found, final guess', low)
