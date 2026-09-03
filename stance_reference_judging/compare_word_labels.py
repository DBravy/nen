"""Agreement between two word-label files (word<TAB>label[<TAB>...]); labels R/S/F/X, case-insensitive."""
import sys, csv
from collections import Counter
def load(p):
    d={}
    for line in open(p):
        parts=line.rstrip("\n").split("\t")
        if len(parts)>=2 and parts[0].strip() and parts[1].strip().lower()!="label":
            l=parts[1].strip().upper()[:1]
            if l in "RSFX": d[parts[0].strip().lower()]=l
    return d
a,b=load(sys.argv[1]),load(sys.argv[2]); common=set(a)&set(b)
m=Counter((a[w],b[w]) for w in common); labs="RSFX"
print(f"{len(common)} words in both files; agreement {sum(m[(l,l)] for l in labs)/len(common):.3f}")
print("rows = "+sys.argv[1]+", cols = "+sys.argv[2]); print("     "+"".join(f"{l:>7s}" for l in labs))
for r in labs: print(f"{r:>4s} "+"".join(f"{m[(r,c)]:7d}" for c in labs))
for pair in [("S","R"),("R","S")]:
    ws=[w for w in common if (a[w],b[w])==pair][:40]; print(f"\n{pair[0]}->{pair[1]} examples: {', '.join(ws)}")
