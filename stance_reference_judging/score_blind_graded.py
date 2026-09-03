"""Score graded blind judgments (blind_tails_graded.md) against the hidden key.

Usage:  python score_blind_graded.py graded_labels.tsv
graded_labels.tsv: one line per item `id<TAB>n_stance<TAB>n_reference`. Per item the stance share is
n_stance/(n_stance+n_reference); items with fewer than 3 counted words are dropped. Prints, per tier,
the mean/median stance share per bank, the event-pooled share, and a Mann-Whitney test.
"""
import sys, csv, os, statistics
here=os.path.dirname(os.path.abspath(__file__))
key={r["id"]:r for r in csv.DictReader(open(os.path.join(here,"blind_tails_key.tsv")),delimiter="\t")}
vals={}
for line in open(sys.argv[1]):
    p=line.strip().split("\t")
    if len(p)>=3 and p[0] in key:
        try: vals[p[0]]=(int(float(p[1])),int(float(p[2])))
        except ValueError: pass
print(f"parsed {len(vals)} of {len(key)} items")
try:
    from scipy.stats import mannwhitneyu
except Exception: mannwhitneyu=None
def report(name,keep,MIN=3):
    d={"UWS":[],"GSS":[]}; pooled={"UWS":[0,0],"GSS":[0,0]}
    for i,(s,r) in vals.items():
        k=key[i]
        if not keep(k) or s+r<MIN: continue
        d[k["bank"]].append(s/(s+r)); pooled[k["bank"]][0]+=s; pooled[k["bank"]][1]+=r
    if min(len(d["UWS"]),len(d["GSS"]))<3: print(f"{name}: too few items"); return
    line=f"{name:40s} n={len(d['UWS'])}/{len(d['GSS'])}  mean share {statistics.mean(d['UWS']):.3f}/{statistics.mean(d['GSS']):.3f}  median {statistics.median(d['UWS']):.3f}/{statistics.median(d['GSS']):.3f}  pooled {pooled['UWS'][0]/max(1,sum(pooled['UWS'])):.3f}/{pooled['GSS'][0]/max(1,sum(pooled['GSS'])):.3f}"
    if mannwhitneyu: line+=f"  MWU p={mannwhitneyu(d['UWS'],d['GSS'],alternative='two-sided').pvalue:.2g}"
    print(line)
report("ALL items",lambda k:True)
report("selective_top100 (selected polarity)",lambda k:k["tier"]=="selective_top100")
report("broad_top15 (both polarities)",lambda k:k["tier"]=="broad_top15")
report("mid_random",lambda k:k["tier"]=="mid_random")
